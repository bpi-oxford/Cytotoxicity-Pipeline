FROM mambaorg/micromamba:focal-cuda-11.8.0
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root
RUN apt update
RUN apt install -y \
    wget \
    unzip

# install python packages
USER $MAMBA_USER
WORKDIR /app/cytotoxicity-pipeline
COPY . /app/cytotoxicity-pipeline
RUN pip install -r requirements.txt

# fiji install
USER root
WORKDIR /app
RUN wget https://downloads.micron.ox.ac.uk/fiji_update/mirrors/fiji-latest/fiji-linux64.zip
RUN unzip /app/fiji-linux64.zip 
RUN rm /app/fiji-linux64.zip
WORKDIR /app/Fiji.app
RUN ./ImageJ-linux64 --headless --ij2 --update add-update-site TrackMateCSVImporter https://sites.imagej.net/TrackMateCSVImporter/
RUN ./ImageJ-linux64 --headless --ij2 --update update
RUN chown -R $MAMBA_USER /app
USER $MAMBA_USER

WORKDIR /app/cytotoxicity-pipeline

EXPOSE 8787
# CMD ["pytest"]