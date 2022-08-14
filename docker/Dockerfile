ARG IMAGE_TAG

FROM cnstark/pytorch:${IMAGE_TAG}

ARG EASYTORCH_VERSION
ENV EASYTORCH_VERSION ${EASYTORCH_VERSION}

ADD . /tmp/easytorch

RUN cd /tmp/easytorch && \
    pip install pip --upgrade && \
    pip install -r requirements.txt && \
    rm -rf .eggs && \
    python setup.py install && \
    rm -rf /tmp/easytorch
