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

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
