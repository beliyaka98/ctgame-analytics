// Jenkins pipeline of ctgame-analytics (declarative syntax, multibranch job).
//
// Every commit on every branch: lint -> unit tests -> Docker image -> the analysis run inside it.
// Commits on main also: push the image to the registry -> deploy the report to the k3s cluster.
// The Jenkins server itself is described in ci/jenkins/ (Docker Compose + configuration as code).

pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '20'))
        disableConcurrentBuilds()
    }

    parameters {
        booleanParam(name: 'PUSH_TO_DOCKERHUB', defaultValue: false,
                     description: 'Also push the image to Docker Hub (needs a "dockerhub" username + access token credential)')
        string(name: 'DOCKERHUB_REPO', defaultValue: 'beliyaka98/ctgame-analytics',
               description: 'Docker Hub repository for the optional push')
    }

    environment {
        IMAGE = 'ctgame-analytics'
        REGISTRY = 'localhost:5000'          // registry service in ci/jenkins/compose.yaml
        NAMESPACE = 'ctgame-test'
        KUBECONFIG = "${WORKSPACE}/.kube/config"
        PIP_DISABLE_PIP_VERSION_CHECK = '1'
    }

    stages {
        stage('Prepare') {
            steps {
                script {
                    env.GIT_SHA = sh(script: 'git rev-parse HEAD', returnStdout: true).trim()
                    env.GIT_SHORT = env.GIT_SHA.take(7)
                    env.VERSION = sh(script: '''sed -n 's/^version = "\\(.*\\)"/\\1/p' pyproject.toml''',
                                     returnStdout: true).trim()
                    env.TAG = "${env.VERSION}-${env.BUILD_NUMBER}-${env.GIT_SHORT}"
                    currentBuild.description = "${env.GIT_SHORT} -> ${env.IMAGE}:${env.TAG}"
                }
                sh 'python3 --version && docker version --format "Docker {{.Server.Version}}" && kubectl version --client'
            }
        }

        stage('Set up Python') {
            steps {
                // The same pinned versions as in the Docker image, plus the test tools.
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --quiet -r requirements/runtime.txt
                    pip install --quiet -e ".[dev]"
                '''
            }
        }

        stage('Static checks') {
            parallel {
                stage('Ruff') {
                    steps {
                        sh '. .venv/bin/activate && ruff check --output-format=concise . && ruff format --check .'
                    }
                }
                stage('Mypy') {
                    steps {
                        sh '. .venv/bin/activate && mypy'
                    }
                }
                stage('Hadolint') {
                    steps {
                        sh 'docker run --rm -i hadolint/hadolint:v2.12.0 < Dockerfile'
                    }
                }
            }
        }

        stage('Unit tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    pytest --junitxml=reports/junit.xml \
                           --cov --cov-report=term --cov-report=xml:reports/coverage.xml
                '''
            }
            post {
                always {
                    junit 'reports/junit.xml'
                    recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
                                   sourceDirectories: [[path: 'src']])
                }
            }
        }

        stage('Build Docker image') {
            steps {
                sh '''
                    docker build \
                        --build-arg VERSION="$VERSION" \
                        --build-arg VCS_REF="$GIT_SHA" \
                        --label "org.opencontainers.image.created=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                        --label "org.jenkins-ci.build-url=$BUILD_URL" \
                        --tag "$IMAGE:$TAG" .
                    docker image ls "$IMAGE:$TAG"
                '''
            }
        }

        stage('Run analysis in the image') {
            steps {
                // Smoke test: the full research pipeline on synthetic data, inside the new image.
                sh '''
                    docker run --rm "$IMAGE:$TAG" --version
                    docker rm -f "smoke-$TAG" >/dev/null 2>&1 || true
                    docker run --name "smoke-$TAG" --entrypoint sh "$IMAGE:$TAG" -c '
                        ctgame simulate --out /work/data &&
                        ctgame evaluate --logs /work/data/logs.csv &&
                        ctgame report --data /work/data --out /work/site'
                    rm -rf site && docker cp "smoke-$TAG:/work/site" site
                    docker rm "smoke-$TAG"
                    test -s site/index.html
                '''
                archiveArtifacts artifacts: 'site/**', fingerprint: true
            }
        }

        stage('Push image') {
            when { branch 'main' }
            steps {
                sh '''
                    for tag in "$TAG" latest; do
                        docker tag "$IMAGE:$TAG" "$REGISTRY/$IMAGE:$tag"
                        docker push --quiet "$REGISTRY/$IMAGE:$tag"
                    done
                '''
            }
        }

        stage('Push to Docker Hub') {
            when {
                branch 'main'
                expression { params.PUSH_TO_DOCKERHUB == true }
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub',
                                                  usernameVariable: 'DOCKERHUB_USER',
                                                  passwordVariable: 'DOCKERHUB_TOKEN')]) {
                    sh '''
                        echo "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USER" --password-stdin
                        for tag in "$TAG" latest; do
                            docker tag "$IMAGE:$TAG" "$DOCKERHUB_REPO:$tag"
                            docker push --quiet "$DOCKERHUB_REPO:$tag"
                        done
                        docker logout
                    '''
                }
            }
        }

        stage('Deploy to k3s') {
            when { branch 'main' }
            steps {
                // k3s writes its kubeconfig to a shared volume; inside the Compose network the
                // API server is reachable as https://k3s:6443.
                sh '''
                    mkdir -p "$(dirname "$KUBECONFIG")"
                    cp /k3s/kubeconfig.yaml "$KUBECONFIG"
                    kubectl config set-cluster default --server=https://k3s:6443 >/dev/null

                    kubectl apply -f deploy/k8s/namespace.yaml
                    sed -e "s|image: ctgame-analytics:latest|image: $REGISTRY/$IMAGE:$TAG|" \
                        -e "s|GIT_COMMIT_PLACEHOLDER|$GIT_SHA|" \
                        -e "s|BUILD_URL_PLACEHOLDER|$BUILD_URL|" \
                        deploy/k8s/deployment.yaml | kubectl apply -f -
                    kubectl apply -f deploy/k8s/service.yaml
                    kubectl -n "$NAMESPACE" annotate deployment/ctgame-report --overwrite \
                        kubernetes.io/change-cause="Jenkins #$BUILD_NUMBER, commit $GIT_SHORT, image $TAG"

                    kubectl -n "$NAMESPACE" rollout status deployment/ctgame-report --timeout=300s
                    kubectl -n "$NAMESPACE" get deployment,pods,service -o wide
                '''
            }
            post {
                failure {
                    sh 'kubectl -n "$NAMESPACE" rollout undo deployment/ctgame-report || true'
                }
            }
        }

        stage('Smoke test deployment') {
            when { branch 'main' }
            steps {
                sh '''
                    deployed=$(kubectl -n "$NAMESPACE" get deployment/ctgame-report \
                        -o jsonpath='{.spec.template.spec.containers[0].image}')
                    echo "deployed image: $deployed"
                    test "$deployed" = "$REGISTRY/$IMAGE:$TAG"

                    served=$(curl -fsS --retry 10 --retry-delay 3 --retry-all-errors http://k3s:30080/version.txt)
                    echo "served commit: $served"
                    test "$served" = "$GIT_SHA"
                    curl -fsS http://k3s:30080/ -o deployed.html
                    grep -q "ctgame" deployed.html
                    kubectl -n "$NAMESPACE" rollout history deployment/ctgame-report
                '''
            }
        }
    }

    post {
        always {
            // Keep the host tidy: the pushed image lives on in the registry and in k3s.
            sh '''
                docker rm -f "smoke-$TAG" >/dev/null 2>&1 || true
                docker image rm "$IMAGE:$TAG" "$REGISTRY/$IMAGE:$TAG" >/dev/null 2>&1 || true
            '''
        }
        success {
            echo "Image ${env.IMAGE}:${env.TAG} passed all stages."
        }
    }
}
