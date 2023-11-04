# Gait Analyzer

Analyze your gait for health disorders at the comfort of your home in your own personal computer.

![A non-gendered humanoid with rainbow colors walking](https://gaitanalyzer.s3.us-east-2.amazonaws.com/logo.png)

## Why

Gait abnormalities can be attributed to various [musculoskeletal and neurological conditions](https://stanfordmedicine25.stanford.edu/the25/gait.html) and so gait analysis is being used as an important diagnostic tool by doctors.

Automated gait analysis requires expensive motion capture or multiple-camera systems. But with Gait Analyzer one can analyze their gait in comfort and privacy of their home on their computer.

## How

Gait Analyzer implements the algorithm published in the paper titled [Automated Gait Analysis Based on a Marker-Free Pose Estimation Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10384445/).

This algorithm for gait analysis is shown to be as reliable as a motion capture system for most scenarios.

Gait Analyzer further uses llama2 large language model to interpret the gait data to the end user in simple terms.

## Video Demo

[![A non-gendered humanoid with rainbow colors walking, Docker and Ollama logos](https://gaitanalyzer.s3.us-east-2.amazonaws.com/Video_Thumbnail.png)](https://www.youtube.com/watch?v=flt7vQNh-fM)
Clicking the above image will open the video on YouTube.

## Features

- Do gait analysis on videos locally on your computer.
- Annotated Video with pose-estimation.
- Distances, Peaks and Minima plotted for each leg.
- Displaying Gait data.
- Download of gait data as .csv file.
- Gait pattern explanation using Large Language Model.

## Screenshots

### Annotated video

![A dwarf person walking from left to right with pose detection annotated on him](https://gaitanalyzer.s3.us-east-2.amazonaws.com/annotated.gif)

### Charts

![Chart showing distances, peaks and minima for left leg](https://gaitanalyzer.s3.us-east-2.amazonaws.com/chart-1.png)

![Chart showing distances, peaks and minima for right leg](https://gaitanalyzer.s3.us-east-2.amazonaws.com/chart-2.png)

### Gait Data

![Gait data in table](https://gaitanalyzer.s3.us-east-2.amazonaws.com/gait-data.png)

### Gait pattern explanation

![LLM generated explanation of gait data -1](https://gaitanalyzer.s3.us-east-2.amazonaws.com/gait-explanation-1.png)
![LLM generated explanation of gait data -2](https://gaitanalyzer.s3.us-east-2.amazonaws.com/gait-explanation-2.png)

## Architecture

![Gait Analyzer Architecture](https://gaitanalyzer.s3.us-east-2.amazonaws.com/gait-analyzer-architecture.png?nocache=true)

## Usage

### Docker

Use Gait Analyzer to analyze your gait on your computer using [Docker](https://hub.docker.com/r/abishekmuthian/gaitanalyzer).

### Setup

Run the LLM model on CPU

```bash
mkdir gaitanalyzer && cd gaitanalyzer
sh -c "$(curl -fsSL https://raw.githubusercontent.com/abishekmuthian/gaitanalyzer/main/install.sh)"
```

Run the LLM model on GPU

_Note: Requires [Nvidia drivers and Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation) to be installed._

```bash
mkdir gaitanalyzer && cd gaitanalyzer
sh -c "$(curl -fsSL https://raw.githubusercontent.com/abishekmuthian/gaitanalyzer/main/install-gpu.sh)"
```

### Thanks & Credits

- Authors of [Automated Gait Analysis Based on a Marker-Free Pose Estimation Model](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10384445/).

- [Mediapipe](https://github.com/google/mediapipe).

- [Ollama](https://github.com/jmorganca/ollama).

### License

Copyright (C) 2023 Abishek Muthian (Gait Analyzer)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
