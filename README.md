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

* Run the player as a background job

```
$ cd home/dan/work/oakes/neon-player
$ <magic virual env activation command here>
$ python -m pupil_labs.neon_player <recording_folder> --job RetinaFaceFaceMapper.detect_and_export <export_folder>
```

In the above examples, _<recording_folder>_ refers to the folder holding the recording (not the recording file). The _<export_folder>_ is a folder where export files (face_positions.csv, gaze_on_face.csv, fixations_on_face.csv) are written.

```
$ python -m pupil_labs.neon_player /home/dan/work/oakes/data/2026-04-28_19-04-55-97e3b2d4 --job RetinaFaceFaceMapper.detect_and_export "/home/dan/work/oakes/export"
```


### Plugin interface

The Neon Player allows us to develop our own code that can be used within the player. A plugin can analyze the scene video frame-by-frame, export the results, and display them in the interactive viewer. 

Plugin files are python files, placed into user's plugin folder. The plugins (and neon player) use a tool called `uv` to update the pytho virtual environment with any packages that the plugin needs. I guess this is useful for a situation like ours - plugin development happens in one place, but you want to install the plugin on other installations of the player. The `uv` tool simplifies the installation of any needed packages. 

#### Dev notes


* Plugin interface methods:

```
def on_recording_loaded(self, recording: NeonRecording) -> None:
```

This is called after the recording is present and loaded. Good time to load any data from cache for the recording.

```
def render(self, painter: QPainter, time_in_recording: int) -> None:
```

Called for each frame, when it is going to be drawn/rendered in the main window. Draw stuff on the scene using Qt's QPainter.

```
def on_disabled(self) -> None:
```

If the "enabled" checkbox is un-checked, the plugin is "disabled", and this is called. Good place to remove any plots from the timeline (that's what other plugins do here). 

* Background job handoff

Background jobs that are set up as an `@action` and connected to a button can run in the interactive viewer. When these jobs run, they run in a separate thread, with a separate instantiation of the Plugin object. Thus, the Plugin's method(s) that run during the background job should NOT save any state. The `finished` slot is called from the main thread, not the background thread, so the object is a separate instantiation. 

The `_detect_all_frames` job is connected to the "Run Detection" button. 

```
@action
def run_detection(self) -> None:
    """Detect faces across the whole recording (runs in background)."""
    job = self.job_manager.run_background_action(
        "RetinaFace Detection",
        "RetinaFaceFaceMapper._detect_all_frames",
    )
    job.finished.connect(self._load_all_from_cache)
```

Here, the method `_detect_all_frames` is called from the background thread. It does the analysis frame-by-frame, and writes the results to the `cache` folder. When that method returns, the job's `finished` slot is called from the main thread. The `_load_all_from_cache` method re-loads that cache info and makes it available to the running player. For example, the data becomes available for use in the `render()` method.



* Get the version numbers right! 

The plugins require some boilerplate code if they will need to install any dependent libraries. In the original version of the plugin, one of the version numbers was wrong:

```
# /// script
# requires-python = ">=3.11"
# # (Neon Player venv detected as Python 3.12 on this machine)
# dependencies = [
#   "insightface>=0.7.3",
#   "onnxruntime>=1.16.0",
#   "opencv-python>=4.8.0",
#   "numpy>=1.24.0",
#   "pl-neon-recording>=0.1.13",
# ]
# ///
```

The last item in the list was originally this:

```
#   "pl-neon-recording>=1.0.0",
```

When the `uv` tool tried to install that library it couldn't - because the pupil labs modules are at a different version. 

## Help

Call me. 

## Authors

Dan Sperka (djsperka@ucdavis.edu)

## Version History

* 0.1
    * Initial Release

