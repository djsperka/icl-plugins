# ICL Plugins for Neon Player

Development files for plugins created for the Neon Player. 

## Description

The initial plugin created performs basic facial recognition on the frames of the scene video. It mimics the functionality of the Neon Cloud's _Face Mapper Enrichment_ in the Neon Player application.

uses https://www.insightface.ai/research/retinaface

## Getting Started

### Dependencies

* This plugin runs within the Neon Player, so it must be installed and working. I installed from source, following directions [here](https://github.com/pupil-labs/neon-player).
* I am using python 3.12.12 in a virtual environment. I am working on Ubuntu 24.04.

### Installing

* Install Neon Player using instructions [here](https://github.com/pupil-labs/neon-player). You can create a virtual environment in any way, just run the `uv` command from within that venv. 
* Clone this repository. 
* Copy the plugin file(s) to the plugin folder in 
 - (linux) `~/Pupil\ Labs/Neon\ Player/plugins`
 - (windows) __??? TBD__

### Executing program

* Move to the player's installation directory
* Activate your virtual env
* Run the player interactively:

```
$ cd home/dan/work/oakes/neon-player
$ <magic virual env activation command here>
$ python -m pupil_labs.neon_player <recording folder, or none - will require you to select one>
```

## Help

Call me. 

## Authors

Dan Sperka (djsperka@ucdavis.edu)

## Version History

* 0.1
    * Initial Release

