# Manure Pyrolysis IAM

This study uses the Global Change Analysis Model (GCAM) integrated assessment model (IAM) to analyze the impact on food, energy, land use, and climate sectors from the introduction of pyrolysis of animal manures in 2050. The project has 2 folders:

    data
    gcam
    xml

## Locations

Global (32 GCAM regions)

## Files overview

The root folder contains the Python scripts used in the study. There are ten Python scripts and three folders. Details of each file or folder is provided below:

- data/: This folder contains three folders
  - data_analysis/: contains excel files for results calculations
    - images/: contains images used in results plotting and in the supplemental information
    - supplementary_tables/: contains supplementary data tables for biochar application rates
  - gcam_out/ contains the extracted data from the gcam xml db. 
    - /released/ contains the results from the reference version of the model. 
    - /<scenario-name>/ contains the results for each named scenario in the sensitivity analysis.  
    - /test/ is a folder for analyzing results from unnamed scenarios as part of the development process
      - The next subfolder level for all 3 extracted models is the RCP pathway
        - /masked/ contains the masked data where years with model errors are removed from data analysis
        - /original/ contains the original data extracted from the GCAM database
        - ref.csv is the output from the database query
        - mask_log.txt contains the list of errors found in the model output
  - maps/ contains map shapefiles for plotting
- gcam/: contains the modified files in the GCAM model. These consist of modifications to R files and input .csv files. Due to their size, the xml config files are not included - instructions for generating them are given below. This is intended to supplement the v7.1 release, and does not contain all necessary files to run the GCAM model. Please install the GCAM model following the instructions in the installation section of this document.
- gcam/input/biochar_land_R/: contains input data files modified for the various scenarios, as well as lists of parameters duplicated in the supplemental information.
- xml/ contains a list of xml queries used to query the GCAM xml db. These only need to be modified if you have a different output folder name.
- check_IO_coef.py checks to ensure that primary and secondary output coefficients are valid to check post-hoc for modeling errors, and remove that data from any subsequent analysis
- constants.py contains a list of constants for use in the project, including the locations for extracting data from the gcam xml db
- data_manipulation.py contains common data manipulation functions for the project
- plotting.py contains code for standard formatting of figures
- plotting_script.py contains the script for processing the data to be plotted and then calling functions from plotting.py to plot the code
- process_data.py is a short script to read data from the gcam xml db and write out .csv files to teh data/gcam_out/ folder
- process_GCAM_data.py splits the single .csv file returned from the gcam xml db and splits it by query
- produce_regional_queries.py converts an .xml file with global queries for the gcam model and makes a query for every region. the gcam xml db does not disaggregate global queries by region
- read_GCAM_DB.py reads data from the gcam xml db.
- supplementary_figures.py conducts additional analysis, much like plotting_script.py

## Requirements

To run the codes in this repository, the following Python and core package versions must be installed:

    pandas~=2.2.1
    geopandas~=0.14.3
    matplotlib~=3.8.3
    numpy~=1.26.4
    plotly~=5.24.1

    Python ~ 3.11
    GCAM model version 7.1

## Installation Guide and Running a Demo

Recommended installation is from the zenodo link here: TBD.

The GCAM model was run on a HP Pavilion Desktop TP01-3xxx using Microsoft Windows 11 Home, with 64GB of RAM. A typical model run will take ~30 minutes to run when using the SSP baseline pathways, expect errors or durations of ~1 day for other SSP/RCP pathways. Model errors in intermediate steps are common.

Users are expected to be proficient in computer software, include the R and Python programming languages, as well as the structure of xml files.

To reproduce the figures in the main manuscript and all data files to produce the figures, run plotting_script.py. Feel free to use the existing data and methods as examples to draw new figures.

To reproduce the GCAM model config xml files, download the GCAM model (http://jgcri.github.io/gcam-doc/index.html). Then, copy over the modified config files contained in this repository to the same location in the JCGRI GCAM project directory.

Then, open the gcam/input/gcamdata/gcamdata.Rproj file, then open gcam/input/biochar_land_R/biochar_land.R file and run the script. 

It may help to have experience building the GCAM project from the R-file (https://jgcri.github.io/gcamdata/articles/getting-started/getting-started.html).

Once the GCAM xml files have been built, run gcam/exe/run-gcam.bat

The expected output will be a ~3GB .xmldb file. This file will need to be named something like "database_basexdb-<scenario-name>_<RCP pathway>", which are required inputs in constants.py, if the data is to processed properly into .csv format for additional analysis.

The .csv output files can be further processed using process_data.py and constants.py to yield the example data in the data folder.

## Citation

Please use the following citation when using the data, methods or results of this work:

TBD

## Overview of Global Change Analysis Model (GCAM)

https://github.com/JGCRI/gcam-core

The Joint Global Change Research Institute (JGCRI) is the home and primary development institution for GCAM, an integrated assessment tool for exploring consequences and responses to global change. Climate change is a global issue that impacts all regions of the world and all sectors of the global economy. Thus, any responses to the threat of climate change, such as policies or international agreements to limit greenhouse gas emissions, can have wide ranging consequences throughout the energy system as well as on land use and land cover. Integrated assessment models endeavor to represent all world regions and all sectors of the economy in an economic framework in order to explore interactions between sectors and understand the potential ramifications of climate mitigation actions.

GCAM has been developed at PNNL for over 20 years and is now a freely available community model and documented online (See below). The team at JGCRI is comprised of economists, engineers, energy experts, forest ecologists, agricultural scientists, and climate system scientists who develop the model and apply it to a range of science and policy questions and work closely with Earth system and ecosystem modelers to integrate the human decision components of GCAM into their analyses.
Model Overview

GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labor productivity growth assumptions drive the energy and land-use systems employing numerous technology options to produce, transform, and provide energy services as well as to produce agriculture and forest products, and to determine land use and land cover. Using a run period extending from 1990 – 2100 at 5 year intervals, GCAM has been used to explore the potential role of emerging energy supply technologies and the greenhouse gas consequences of specific policy measures or energy technology adoption including; CO2 capture and storage, bioenergy, hydrogen systems, nuclear energy, renewable energy technology, and energy use technology in buildings, industry and the transportation sectors. GCAM is an Representative Concentration Pathway (RCP)-class model. This means it can be used to simulate scenarios, policies, and emission targets from various sources including the Intergovernmental Panel on Climate Change (IPCC). Output includes projections of future energy supply and demand and the resulting greenhouse gas emissions, radiative forcing and climate effects of 16 greenhouse gases, aerosols and short-lived species at 0.5×0.5 degree resolution, contingent on assumptions about future population, economy, technology, and climate mitigation policy.

“The Global Change Analysis Model (GCAM) is a multisector model developed and maintained at the Pacific Northwest National Laboratory’s Joint Global Change Research Institute (JGCRI, 2023) <include additional citations to previous GCAM studies as relevant>. GCAM is an open-source community model. In this study, we use GCAM v NN. The documentation of the model is available at the GCAM documentation page (http://jgcri.github.io/gcam-doc) and the description below is a summary. GCAM includes representations of: economy, energy, agriculture, and water supply in 32 geopolitical regions across the globe; their GHG and air pollutant emissions and global GHG concentrations, radiative forcing, and temperature change; and the associated land allocation, water use, and agriculture production across 384 land sub-regions and 235 water basins.
JGCRI, 2023. GCAM Documentation (Version 7.0). https://github.com/JGCRI/gcam-doc. Joint Global Change Research Institute. https://zenodo.org/doi/10.5281/zenodo.11377813.
