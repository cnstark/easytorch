
EASYTORCH_VERSION=`python -c "from easytorch import __version__
print(__version__)"`

docker build \
    --build-arg IMAGE_TAG=${IMAGE_TAG} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t cnstark/pytorch:${EASYTORCH_VERSION}-${IMAGE_TAG} \
    -f docker/Dockerfile \
    .
