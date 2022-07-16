FROM python:3.9-slim

RUN groupadd -r predictor && useradd -m --no-log-init -r -g predictor predictor

# Your code and weights are in /opt/code, images are in /input, predictions are written to /output
RUN mkdir -p /opt/code /input /output \
    && chown predictor:predictor /opt/code /input /output

USER predictor
WORKDIR /opt/code

ENV PATH="/home/predictor/.local/bin:${PATH}"

# Install pip
RUN python -m pip install --user -U pip

# Copy all your needed files and directories to /opt/code
COPY --chown=predictor:predictor evaluation /opt/code/evaluation
COPY --chown=predictor:predictor models /opt/code/models
COPY --chown=predictor:predictor utils /opt/code/utils
COPY --chown=predictor:predictor weights /opt/code/weights

COPY --chown=predictor:predictor requirements.txt /opt/code/
COPY --chown=predictor:predictor predict.py /opt/code/
COPY --chown=predictor:predictor Demo_densebiasnet.py /opt/code/

# Install packages specified in requirements.txt
RUN python -m pip install --user --no-cache-dir -r requirements.txt

# Entry
CMD python3 predict.py