# Basic Trimmer - The Golden Lab 

The Basic Trimmer is a tool to aid the use of Simple Behavioral Analysis (SimBA) through data preparation.

A major technical limitation in the study of complex social behavior of freely moving rodents is the bottleneck caused by manual annotation of behavior. Behavioral annotation can be subjective, extremely time-intensive, and prone to observer drift. Simple Behavioral Analysis (SimBA), an open-source package, uses machine learning (ML) approaches to automate behavioral predictions through the combination of pose estimation and supervised ML. Individual experiments often incorporate thousands of video recordings, all of which need to be preprocessed and maintained; this includes locating individual experimental trials, identifying behavioral events within each trial, and choosing trimming points to focus on specific behavioral epochs. Manual preprocessing and project management requires unfeasible effort, often yielding video datasets with inaccuracies in timing or content, leading to inaccurate ML predictive classifications.

The Basic Trimmer allows the user to navigate through all of the start/end points that they would like to trim and handles the trimming process for them. Currently it assumes and accomodates up to 12 trials, based on operant self-administration tasks developed at the Golden Lab, but it should be *relatively easily* extensible to tasks with 12 or fewer trials.  

<br> <br>

For more information on usage please refer to the Trimmer's [readthedocs](https://trimmer-golden-lab.readthedocs.io/en/latest/index.html)  

**Example workflow (GUI view out of date, but general functionality is similar)**
![Full Trimmer Process Gif (outdated view)](https://github.com/virginiavw/Trimmer/raw/86d55e09d7f27e50bddf08f25eda51fb78636b23/docs/source/images/fullprocess.gif)  
<br>
  
  
## Features added / in development
- [x] Manually or automatically add trim (clip) start and end points for behavioral trials
- [x] Set multiple clip trim points per trial, with unique labels set individually for each trial, or for convenience using index labeling (0, 1, 2)
- [x] Save trim points for each video for later clipping
- [x] Simple FFMPEG-based script to batch process videos using saved trim points
- [x] Output folder-specific metadata file to track processed videos and clips created
- [x] Configurable trial length and trial-to-trial gap for automatic settings of trim points
- [x] Save/load preferences for inputs
- [x] Quickly add ITI clips (periods between trial trim points)
- [x] Extended logging
- [ ] Streamlined preset labeling (with and without indexing) for auto-labels
- [ ] Configurable number of trials
- [ ] Docs / tutorial overhaul for newly added features

<br> <br> 

## Authors
*   **Kevin N. Schneider**
    *   GitHub: [@kevsch88](https://github.com/kevsch88)
    *   Email: kevsch88@gmail.com
*   **Virginia Wang**
    *   GitHub: [@virginiavw](https://github.com/virginiavw)

## Prerequisites

Currently tested working only on Windows (Aug 2025).

**UV package manager**  
At the moment environment setup is recommended via the [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager for streamlined setup, which uses the `pyproject.toml` file.

**FFMPEG**  
For video clipping/trimming.


## Installation

Clone the repo
```
git clone https://github.com/sgoldenlab/Trimmer.git
cd Trimmer
```

After uv is installed, the environment can be quickly installed from within the with:  
`uv sync`

Once complete, the trimmer GUI should launch with
`uv run basic_trimmer`
