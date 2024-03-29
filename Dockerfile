## Use rstudio installs binaries from RStudio's RSPM service by default,
## Uses the latest stable ubuntu, R and Bioconductor versions
FROM  bioconductor/bioconductor_docker:3.17

#install R CRAN binary packages
RUN install2.r -e \
testthat

## Install remaining packages from source
COPY ./requirements-src.R .
RUN Rscript requirements-src.R

## Install Bioconductor packages
COPY ./requirements-bioc.R .
RUN Rscript -e 'requireNamespace("BiocManager"); BiocManager::install(ask=F);' \
&& Rscript requirements-bioc.R

## Install R packages from github
COPY ./requirements-github.R .
RUN Rscript requirements-github.R

## Add packages dependencies for python
RUN apt-get update \
        && apt-get install -y --no-install-recommends apt-utils \
        && apt-get install -y --no-install-recommends \
        ## Basic deps
        python3-pip \
        python3-dev \
        python3-setuptools

## Install Python requirements
COPY ./requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install wheel setuptools
RUN pip3 install -r requirements.txt

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy over project
COPY . .

# Run tests
RUN python3 -m unittest discover -s unittests