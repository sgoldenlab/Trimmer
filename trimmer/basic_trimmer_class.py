# Yo hello future me when you get back, change the disabled = true bits into the actual version and put it
# on both computers. also disable/reable?able? the GOTO buttons thanks!

# %% Imports
import pdb
import cv2
import pandas as pd
import numpy as np
import os
import PySimpleGUI as sg
from datetime import datetime 
from time import gmtime, strftime, time, sleep
import imutils
from imutils.video import FileVideoStream
# import sys    currently unused
# import threading
import yaml

# %% Setup
#src_fold = r'./'  # init source folder"
# video file extensions
file_exts_accepted = '.mp4'

# options
debug = False
class trim_list: 
    def __init__(self, trial_num):
        self.trim_starts = list()
        self.trim_ends = list()
        self.trim_labels = list()
        self.trim_pts_array = np.ndarray((trial_num,3))
        self.trim_pts_labels = np.ndarray((trial_num,2), dtype='object')
        self.trim_pts_array[:] = np.nan
        print("TRIM ARRAY", self.trim_pts_array)
        print("TRIM LABELS", self.trim_pts_labels)
        self.printout = list() # will output list of trimpoints added to be saved
    
    def add_trim_points(self, trial_idx, trim_start, trim_end, trim_label):
        if any([np.isnan(ix) for ix in self.trim_pts_array[:,0]]):    # if any uninitialized rows
            nan_idx = np.where([np.isnan(ix) for ix in self.trim_pts_array[:,0]])[0][0]  # get first index where above is true
            self.trim_pts_array[nan_idx] = np.array([[trial_idx, trim_start, trim_end]])     # replace nan values if any
            self.trim_pts_labels[nan_idx] = np.array([[trial_idx, trim_label]]) 
        else:
            self.trim_pts_array = np.append(self.trim_pts_array, np.array([[trial_idx, trim_start, trim_end]]), axis=0)
            self.trim_pts_labels = np.append(self.trim_pts_labels, np.array([[trial_idx, trim_label]]), axis=0)

        sort_idx = self.trim_pts_array[:,0].argsort()
        self.trim_pts_array = self.trim_pts_array[sort_idx]
        self.trim_pts_labels = self.trim_pts_labels[sort_idx]
        
    def get_printed_trimpoints_listbox(self, fps:int) -> list:
        self.printout = list()
        for i,t in enumerate(self.trim_pts_array[self.trim_pts_array[:,0].argsort()]):
            try:
                self.printout.append(f'trial {int(t[0])+1}, "{self.trim_pts_labels[i,1]}": {self.frame_to_time(t[1], fps)}-{self.frame_to_time(t[2], fps)}')
            except ValueError as err:
                # printout.append([f'trial {t[0]}: {t[1]}-{t[2]}'])
                continue
    
    def remove_selected_tp(self, listbox_element: sg.Listbox):
        to_remove = listbox_element.get()
        if to_remove is not []:
            for item in to_remove:
                # remove_idx = listbox_element.get_list_values().index(item)
                remove_idx = self.printout.index(item)
                self.trim_pts_array = np.delete(self.trim_pts_array, remove_idx, axis=0)
                self.trim_pts_labels = np.delete(self.trim_pts_labels, remove_idx, axis=0)
                self.printout.remove(item)   # remove and then update list *** sus
        else:
            print('Tried to remove trimpoints but none were selected.')
    
    def frame_to_time(self, frame, fps, time_format="%M:%S"):
        if isinstance(frame, list):
            return [strftime(time_format, gmtime(round(f / fps))) for f in frame]
        else:
            return strftime(time_format, gmtime(round(frame / fps)))
        
class vid_info: 
    def __init__(self, fps, tot_frames, height, width, trial_num, cur_frame, ret, frame):
        self.fps = fps
        self.tot_frames = tot_frames
        self.height = height
        self.width = width
        self.trial_num = trial_num
        self.cur_frame = cur_frame
        self.ret = ret
        self.frame = frame

# %% Functions

def place_noex(elem):   # places pysimplegui element without expanding its horizontal (x) reach
    """
    Places pysimplegui element without expanding its horizontal (x) reach
    :param elem: PySimpleGUI element, ex: place_noex(sg.Button(...))
    :return:
    """
    return sg.Column([[elem]], pad=(0, 0))


def place(elem):    # places pysimplegui element and expands its horizontal (x) reach
    """
    Places pysimplegui element and expands its horizontal (x) reach
    :param elem: PySimpleGUI element, ex: place_noex(sg.Button(...))
    :return:
    """
    return sg.Column([[elem]], pad=(0, 0), expand_x=True)

def buildWindow(debug=False):  # TODO: disable text boxes, buttons etc as relevant
    """
    update fileName, filePath, Slider(tot_frames), info column, checkboxes (default=next(reward), text_color=next(r_col))
    """

    top_region = [[sg.Txt('Basic Trimmer (WIP)', font='Arial 10', justification='left', pad=((0,20),0)),
                #    sg.Column('', expand_x=True),
                    sg.Text('Videos Folder:', font='Arial 10',pad=((5,0),0)), 
                    sg.In(size=(40,1), enable_events=True ,key='-FOLDERI-', pad=((1,5),0)), 
                    sg.FolderBrowse(initial_folder='./', font='Arial 10', pad=((0,25),0)),
                    sg.Text('Output Folder:', font='Arial 10', pad=(0,0)), 
                    sg.In('', size=(40,1), enable_events=True ,key='-FOLDERO-', pad=((1,5),0)),
                    sg.Push(), place_noex(sg.B('Help', size=(7, 1), font='Arial 10', k='bHelp'))],
                  [sg.Txt('Video name: ', key = 'fileName', font='Arial 12')],
                  [sg.Txt('Video filepath: ', key = 'filePath', expand_x=False, font='Arial 9')]
                  ]
    # TODO: flexible resizing of window output, 
    #   some image elem properties will dictate a good chunk of that
    # left of GUI, shows frames and contains scrollbar and buttons to interact with the playback
    vid_column = [[sg.Column([[sg.Image(key='-IMAGE-', expand_x=False, expand_y=False, size=(640,480))]],
                             element_justification='center', expand_x=True)],
                  [sg.T('', font='Arial 10', justification='left')],
                  [sg.Text('', font='Arial 8', k='sleft', size=(8, 1), pad=(0,(20,0))),
                   sg.Slider(range=(0, 1), expand_x=True, orientation='h', key='-SLIDER-',
                             resolution=1, enable_events = True),
                   sg.T('', font='Arial 8', k='sright', size=(8,1), pad=(0,(20,0)))],
                  [sg.B('Run', key='-RUN-PAUSE-', font='Arial 10',  pad=(5, 0), auto_size_button=True),
                   sg.T('Frame step:', auto_size_text=True, pad=((25,0),0)),
                   place_noex(sg.Button('-30', font='Arial 10', k='bPrevF30', auto_size_button=True, pad=((0,5),1))),
                   place_noex(sg.Button('-5', font='Arial 10', k='bPrevF5', auto_size_button=True)),
                   place_noex(sg.Button('-1', font='Arial 10', k='bPrevF', auto_size_button=True)),
                   place_noex(sg.Button('+1', font='Arial 10', k='bNextF', auto_size_button=True)),
                   place_noex(sg.Button('+5', font='Arial 10', k='bNextF5', auto_size_button=True)),
                   place_noex(sg.Button('+30', font='Arial 10', k='bNextF30', auto_size_button=True)),
                   place_noex(sg.Input(key='-INPUT-TIME-', size=(5, 1))),
                   place_noex(sg.Button('Confirm', key='-CONFIRM-INPUT-')),
                   place_noex(sg.Button('Back', key = '-back-')),
                   #sg.Push(),
                   place_noex(sg.Button('Reset', font='Arial 10', k='bReset', auto_size_button=True)),
                   place_noex(sg.Button('Check F', font = 'Arial 10', k = 'bFrames', auto_size_button=True,
                                        disabled=False if debug else True)) 
                                        if debug else 
                                        place_noex(sg.Txt('', font = 'Arial 10', size=(5,1))),
                   place_noex(sg.Button('Next Video', font='Arial 10', k='bNV', auto_size_button=True))],
                  
                  ]
    right_test = [[sg.Column([
                            [sg.Txt('Property', size=(12,1), font='Arial 12', justification='left', pad=(0, 0)),
                            sg.Txt('Value', size=(12, 1), font='Arial 12', justification='left', pad=(0, 0))],
                            [sg.Txt('Mouse ID', size=(13,1),font='Arial 10', justification='left', pad=(0, 5)),
                            sg.I('mouse01', key = '-mouseID-',  text_color = 'gray', enable_events=True,
                                    font='Verdana 10', size=(20, 1), pad=(0, 5), expand_x=True)],
                            [sg.Txt('Project ID', size=(13,1),font='Arial 10', justification='left', pad=(0, 5)),
                            sg.I('project01', key = '-pID-',text_color = 'gray', enable_events=True,
                                    font='Verdana 10', size=(20, 1), pad=(0, 5), expand_x=True)],
                            [sg.Txt('Session/Day',size=(13,1), font='Arial 10', justification='left', pad=(0, 5)),
                            sg.I('D01', key = '-sess-', text_color = 'gray',  enable_events=True,
                                    font='Verdana 10', size=(20, 1), pad=(0, 5), expand_x=True)],
                            [sg.Txt('Date (YYMMDD)', size=(13,1),font='Arial 10', justification='left', pad=(0, 5)),
                            sg.I(f'{datetime.today().strftime("%y%m%d")}', 
                                    key = '-date-',  text_color = 'gray',  enable_events=True,
                                    font='Verdana 10', size=(20, 1), pad=(0, 5), expand_x=True)],
                            [sg.Txt('User initials', size=(13,1),font='Arial 10', justification='left', pad=(0, 5)),
                            sg.I('', key = '-user-',  text_color = 'gray', enable_events=True,
                                    font='Verdana 10', size=(20, 1), pad=(0, 5), expand_x=True)],
                            [sg.Txt('Preview output prefix:', auto_size_text=True, font='Arial 10', 
                                     justification='left', pad=(0, 5))],
                             [sg.T('', key='-preview-', auto_size_text=True, font='Arial 12', 
                                    text_color='blue', justification='left', pad=(0, 5))]
                            ], vertical_alignment = 'top', expand_x=True, size = (200, 250))],
                   [sg.HorizontalSeparator(pad=(5, 10))],
                   [sg.Column([[sg.Text('Notes:', font='Arial 10',
                                        auto_size_text=True, justification='Left')],
                                [sg.Multiline(font='Arial 10', k='-notes-',
                                        background_color='white',size = (30, 5),  expand_x=True, expand_y=True)],]), 
                    sg.VerticalSeparator(),
                    sg.Column([[sg.Input('', key = '-input-resize-', font='Arial 10', size=(5, 1))],
                                [sg.B('resize', key = '-resize-')],]),],
                   [sg.HorizontalSeparator(pad=(5, 10))],
                   [sg.Column([[sg.Text('Output:', font='Arial 10',auto_size_text=True, justification='Left')],
                               [sg.Output(font='Arial 8', k='-log-',
                       background_color='white', size = (50,15), expand_x=True, expand_y=True)],])]                     
                  ]
                        # sg.Column([[sg.Text('Notes:', font='Arial 10',
                    #                     auto_size_text=True, justification='Left')],
                    #             [sg.Multiline(font='Arial 10', size=(15,10), k='-notes-',
                    #                     background_color='white',  expand_x=True, expand_y=True)],
                    #             ],expand_y=True, expand_x=True, vertical_alignment='bottom', pad=(0,0)),     


    # bottom right of GUI, shows trim-points extracted from video, reward info for each trial, and excluded trials after pressing the GTP button
    # Also shows got to buttons, which can be used to locate a the start of a particular trial, and select frame buttons, which can be used to update the start time of a trial.
    # and exit buttons
    trim_col = [ 
                [
                #sg.Text('', size=(6, 1), pad=(1,0)),
                sg.B('Get trial timepoints', k='-gtp-', expand_x=False, size=(20,1), font='Arial 12', pad=(0, (0,5))),
                sg.B('Add all', font='Arial 12', k='-add_all-', auto_size_button=True, pad=(12, (0,5))),
                sg.B('Clear all', font='Arial 12', k='-clr-', auto_size_button=True, pad=(0, (0,5)))],
                [
                    sg.T('Trial #', font='Arial 10', size=(9, 1), pad=(0,0)), #sg.T('',size=(5,1)),
                    sg.T('Trim start', font='Arial 10', size=(9, 1), justification='center', pad=((0,9),0)),
                    sg.T('Trim end', font='Arial 10', size=(9,1), justification='left', pad=((0,9),0)),
                    sg.T('Trim Point Label:', font='Arial 8', 
                         size=(10,2), justification='center', pad=((0,25),(0,0))),
                    # sg.T('Exclude?', font='Arial 8', size=(7, 1), justification='left', expand_x=False),
                ]
            ]

    #normally -> default=next(reward), text_color=next(r_col)
    #iterator should still work in for loop though...
    global trial_list
    trial_list = list()
    for i in range(1,13):
        trim_col += [sg.T(f'Trial {i}', font='Arial 8', size=(6, 1), justification='left'),
                    # sg.B('Select Frame', font='Arial 7', k=f'gst{i}', size=(5,1)),
                    sg.B('GOTO', font='Arial 7', k=f't{i}_gstart', size=(5,1)),
                    sg.T('', k=f't{i}_start', font='Arial 8', relief='groove', background_color='white', size=(8, 1), pad=(0, 1), enable_events=True),
                    sg.T('', k=f't{i}_end', font='Arial 8', relief='groove', background_color='white', size=(8, 1), pad=(0, 1), enable_events=True),
                    sg.B('GOTO', font='Arial 7', k=f't{i}_gend', size=(5,1), pad=((5,10),1)),
                    sg.Input('', k=f't{i}_label', expand_x=True, font='Arial 8', size=(5, 1), pad=(0, 1)),
                    sg.B('+', k=f't{i}_ADD_TRIMPTS', expand_x=False, font='Arial 8', pad=((0,5), (0,1)), size=(2,1)),
                    # sg.CB('', key=f'-ex{i}-', default=False, disabled=True, enable_events=True)
                    ],
        trial_list.append(f't{i}')


    # top right of GUI, has animal and session info from behavior file
    # incorporates trim column layout for ease when constructing window
    info_column = [[sg.Column(trim_col, pad=(0, 0), vertical_alignment='bottom', size = (350,375))],
                   [sg.Column([[sg.Text('Trim points:', font='Arial 10',
                                    auto_size_text=True, justification='Left')],
                                ### none-selectable version:
                                # [sg.Multiline(font='Arial 10', size=(20,10), k='-tpts_out-',
                                #              background_color='white',  expand_x=True)]
                                # ],expand_y=True, expand_x=True, vertical_alignment='bottom', pad=(0,0))
                                [sg.Listbox(font='Arial 10', size=(45,10), k='-tpts_out-', values=['No trim points added.'],
                                            select_mode='multiple', enable_events=True,
                                            background_color='white',  expand_x=True)],
                                [sg.B('Remove trimpoints', font='Arial 10', k='-remove_tp-', 
                                            auto_size_button=True), sg.Push(), place_noex(sg.B('Save', font='Arial 10', k='-save-', auto_size_button=True))],
                                [sg.B('Clear trimpoints', font='Arial 10', k='-clear_tp-', 
                                           auto_size_button=True),sg.Push(), place_noex(sg.B('Exit', font='Arial 10', auto_size_button=True))
                                ]], expand_y=True, expand_x=True, vertical_alignment='bottom', pad=(0,0), size = (350,None)),                  
                     ]
                   ]

    # combines previous layouts for ease in constructing final layout
    video_layout = [[sg.Column(right_test, k='vid_column', expand_x=False),
                     sg.Column(vid_column, k='vid_column', expand_x=False),
                     sg.VerticalSeparator(pad=(5, 5)),
                     sg.Column(info_column, k='info_column', expand_x=True, expand_y=True)]
                    ]

    # final layout
    layout = [[sg.Column(top_region, k='top_region', expand_x=True)],
    [sg.Column(video_layout, k='video_layout', expand_x=True, expand_y= True)]
    ]
    window = sg.Window('Basic Trimmer', layout, finalize=True, enable_close_attempted_event=True,
resizable=True)
    return window

def close_vid(vid_file: FileVideoStream):
    """Closes video stream out.

    Parameters
    ----------
    vid_file : FileVideoStream
        Currently open video stream.
    """
    vid_file.stream.release()
    vid_file.stop()
    sleep(2)
    print('Video closed')

def load_metadata(filepath: str, debug=False) -> dict or None:
    """Checks if metadata exists and loads it to dictionary.
    Returns None if no metadata found.

    Parameters
    ----------
    filepath : str
        Filepath of yaml file with metadata.
    debug : bool, optional
        Debug printing, by default False

    Returns
    -------
    dict or None
        Dictionary with key:value pairs defined in the yaml string imported.
        `safe_load()` prevents numerical values from being read as string.
    """
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            yaml_dict = yaml.safe_load(file)#, yaml.BaseLoader)
        return yaml_dict
    else:
        if debug:
            print(f'Could not find {os.path.basename(filepath)}.')
        return None


def save_metadata(data: dict, out_filepath: str, debug=False):
    """Outputs metadata to yaml file.

    Parameters
    ----------
    data : dict
        Dict with metadata to export.
    out_filepath : str
        Output filepath for yaml file.
    debug : bool, optional
        Debug printing, by default False
    """
    with open(out_filepath,'w') as file:
                yaml.dump(data, file)
    if debug:
        print(f'Exported metadata to {out_filepath}.')


def scan_video_folder(source_dir: str, out_dir: str, usable_extensions=['.mp4'], debug=False):
    """Scans source folder for videos, checks metadata for processed videos, sets up for trimming.
    Output `video_list` skips videos found in the metadata file, which are put into the list `processed_videos`.

    Parameters
    ----------
    source_dir : str
        Path to folder with videos to trim.
    out_dir : str
        Path to folder where metadata and output files will be stored.
    usable_extensions : list, optional
        List of video format extensions to be included from source_dir, by default ['.mp4']
    debug : bool, optional
        Debug printing, by default False

    Returns
    -------
    video_list: list[str] - list of video filenames to process from source_dir.
    processed_videos: list - list of videos already present in metadata (ie, already processed).
         
    """
    processed_videos = list()
    metadata = load_metadata(os.path.join(out_dir, 'metadata.yaml'))
    if metadata is not None:
        processed_videos = list(metadata.keys())    # keys will be the video filenames
        print('Videos already in metadata:')
        for v in processed_videos:
            print('\t', v)
    else:
        print(f'No metadata found for videos in chosen folder.\n')
    video_list = [video for video in os.listdir(source_dir) if video.endswith(usable_extensions) and video not in processed_videos]
    return video_list, processed_videos


def reset_info(window: sg.Window, filename:str, filepath:str, out_folder='./clips') -> sg.Window:
    """Resets basic window properties and loads in info from metadata file if found.

    Parameters
    ----------
    window : sg.Window
        Currently open PySimpleGUI window.
    filename : str
        Name of video file.
    filepath : str
        Path to video file.
    out_folder : str, optional
        Output folder, by default './clips'

    Returns
    -------
    window: sg.Window
    """
    # # update fileName, filePath and reset slider
    window['fileName'].update(filename)  # - .mp4
    window['filePath'].update(filepath)
    window['-SLIDER-'].update(0) #reset slider
    # window['-mouseID-'].update('mouse01')
    # window['-pID-'].update('project01')
    # window['-sess-'].update('D01')
    # window['-date-'].update(f'{datetime.today().strftime("%y%m%d")}')

    # if metadata, reload last values
    metadata = load_metadata(os.path.join(out_folder, 'metadata.yaml'))
    if metadata:
        # update info column with last entries from metadata
        last_entry = list(metadata.keys())[-1]
        info = metadata[last_entry]['experiment_info']
        window['-mouseID-'].update(value=info['mouse'])
        window['-pID-'].update(value=info['experiment'])
        window['-sess-'].update(value=info['session'])
        print('loaded info from last entry in matadata file')
    return window

def popup (warning : str, type : str = None):
    if type == 'yesno':
        return sg.popup_yes_no(warning, title = 'Warning') == 'Yes'
    elif type == 'error':
        return sg.popup_error(warning)
    else:
        return sg.popup(warning)

# %% Main

def main(debug=False):  
    sg.theme('Dark Blue 3')
    
    #1 ---------------- build layout/initialize variable 
    window = buildWindow(debug=debug)
    vidFile = None
    paused = True   # start video paused
    stop = False    # used to stop loop for some events
    timeout = None  #   None until vid loaded, then 1000/frames per sec rate to match video play
    back = 0 
    # go through and bind info input boxes to focus in/out (click on, out of), for clearing default text
    #   this may go, depending on how useful it is / useful the defaults are
    inputs = ['-mouseID-', '-pID-', '-sess-', '-date-', '-user-']
    input_focusin = list()
    input_focusout = list()
    input_defaults = list(['']*len(inputs))
    resize_factor = 1
    vid = None
    t = None
    # sg.set_options(scaling=5)
    #2 ---------------- open window :))))
    while True:
        
        image_elem = window['-IMAGE-']
        slider_elem = window['-SLIDER-']  
        listbox_elem = window['-tpts_out-']  # listbox with trim points to be exported, if any were added
        event, values = window.read(timeout = timeout)
        
        if stop:
            close_vid(vidFile)
            
        if event == "Exit" or event == sg.WINDOW_CLOSE_ATTEMPTED_EVENT:
            if not popup("Do you really want to exit?\n(Have you saved your work?)", 'yesno'):
                continue
            break
        
        elif event == 'bHelp':
            popup('Well hello my friend!\nSo eventually this button will hopefully lead you to some fun documentation and stuff, '\
                'but for now just ask Virginia or Kevin if you need help! (I hear there is a fun powerpoint you might be sent :) )\n\nThanks my friend.')
            
        if event == '-SLIDER-' and timeout == None: # is this for slider event at end of video?
            timeout = 1000//fps
        
        # if source folder chosen, scan and update
        if event == '-FOLDERI-' and timeout == None:
            folder_in = values['-FOLDERI-']
            if debug:
                print('Videos folder: ', folder_in)
            if values['-FOLDERO-'] == '':   # provide default output folder if none given
                folder_out = os.path.join(folder_in, 'clips')
                window['-FOLDERO-'].update(value=os.path.abspath(folder_out))
            else:
                folder_out = values['-FOLDERO-']
            # scan source folder for videos and metadata, skip videos already in metadata
            vflist, vdone = scan_video_folder(folder_in, folder_out, file_exts_accepted, debug)
            vfiter = iter(vflist)
            os.chdir(folder_in)  # probably not needed, since we want to work with absolute paths or relative to source
            window['bNV'].click() # autoclicks next video; start with first video loaded in 
            
        #Next Video => load next video, update info, reset variables
        if event == 'bNV':
            # update video information
            # -----------cur, just updates with file in folder; change to next video information
            # TODO: Load in info from metadata if present that could be relevant for reuse, eg user.
            if values['-FOLDERI-'] == '':
                print('No source folder selected, select one under "Videos Folder" and try again.')
                continue

            if (vidFile is not None) and (vidFile.running()):   # if a video is still open/running
                print('closing video')
                close_vid(vidFile)
                sleep(3)
            try:
                vidf = next(vfiter)
            except StopIteration as err:
                print('No more unprocessed videos found. Please press "Exit."')
                continue
            
            if vidf in vdone:
                next(vfiter)
            fileName = vidf  # 'SA_OF01_EF1_B7_C1_D4_20220321_10h18.mp4'
            fileRelPath = os.path.join(folder_in, fileName)
            filePath = os.path.abspath(fileRelPath)
            
            # reset and load some info from metadata if present
            reset_info(window, fileName, filePath, out_folder=folder_out)
            # cur_frame = 0

            #load next video 
            vidFile = FileVideoStream(filePath, queue_size=200)
            sleep(2.0)  # give time to load video
            vidFile.start()
            sleep(8.0)  # give time for thread to start up
            
            #reset variables with video information
            tot_frames = vidFile.stream.get(cv2.CAP_PROP_FRAME_COUNT)   
            ret, frame = vidFile.stream.read()
            frame = imutils.resize(frame, width = 1000)
            height, width = vidFile.stream.get(cv2.CAP_PROP_FRAME_HEIGHT), vidFile.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            print('Height: ',  height , ' Width: ' , width)
            print('my screen size; ', sg.Window.get_screen_size())
            fps = round(vidFile.stream.get(cv2.CAP_PROP_FPS))
            #window['sright'].update(f'{strftime("%M:%S", gmtime(num_frames / fps))}')
            trial_num = 12
            vid = vid_info(fps, tot_frames, height, width, trial_num, 0, ret, frame)
            print('Video FPS: ', vid.fps)
            timeout = 1000 // vid.fps
            t = trim_list(trial_num)
            
            #reset slider, image 
            slider_elem.update(0, range = (0, vid.tot_frames))   
            vid.frame = cv2.resize(vid.frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
  
            imgbytes = cv2.imencode('.ppm', vid.frame)[1].tobytes()
            image_elem.update(data=imgbytes)
            for input in inputs:
                input_focusin.append(f'{input}_InFocus')
                window[input].bind('<FocusIn>', '_InFocus')
                input_focusout.append(f'{input}_OutFocus')
                window[input].bind('<FocusOut>', '_OutFocus')
                dict.update({input: ''})
        
        # if just started program, must click "Next Video"
        elif vidFile == None:
            print('Please choose a video folder')
            continue 

        if event == 'bFrames':  # debug=True dependent
            print('Current Frame: ', vid.cur_frame)
            print('num frames: ', vid.tot_frames)
            print('timeout: ', timeout)
            
        # cur_frame runs ahead of the last loaded image, 
        #   so if previous frame was the end frame, can't move forward
        if vid.cur_frame - 1 == vid.tot_frames and event == 'bNextF': 
            print('Video has ended. Please click \"Next Video\"')
            continue

        # if end frame within 30 frames, give warning
        elif vid.cur_frame + 31 > vid.tot_frames and event == 'bNextF30':
            print('Less than 30 frames are left.')
            continue

        # build preview for info text boxes
        preview_str = list()
        for input in inputs[:-1]:
            preview_str.append(values[input])   
        window['-preview-'].update(value='_'.join(preview_str)) # combine string for output based on info values

        # clear inputs when 'focused' on
        if event in input_focusin:           
            input_elem = event.split('_')[0]
            input_defaults[inputs.index(input_elem)] = values[input_elem] # temporary workaround to saving prev text
            window[input_elem].update(value='')
            window[input_elem].update(text_color='black')

        elif event in input_focusout:   # not bad though, currently leaves last input if unchanged
            input_elem = event.split('_')[0]
            if values[input_elem] == '':    # if no text typed in, go back to previous value
                window[input_elem].update(value=input_defaults[inputs.index(input_elem)])
            else:
                window[input_elem].unbind('<FocusIn>')
                if debug:
                    print(f'unbound {input_elem} from focusin')
            window.refresh()

        # TODO: should make func for cur_frame updates, since the same elements tend to need updating in turn
        if int(values['-SLIDER-']) != vid.cur_frame - 1:    
            vid.cur_frame = int(values['-SLIDER-'])
            #vidFile.stream.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)  # set new frame
            slider_elem.update(vid.cur_frame)                           # update scrollbar position and
            # TODO: there's gotta be a better way to line sleft and sright text with the scrollbar,
            #   currently just padding from above to bring them down a bit
            window['sleft'].update(                                 #   relevant surrounding values
                f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')
            if vid.ret: #may not be needed   
                vid.frame = cv2.resize(frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
                try:
                    imgbytes = cv2.imencode('.ppm', vid.frame)[1].tobytes()
                except:
                    imgbytes = cv2.imencode('.png', vid.frame)[1].tobytes()
                    print('tried png')
            image_elem.update(data=imgbytes)  # update shown frame

        slider_elem.update(vid.cur_frame)
        window['sleft'].update(f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')

        # if not paused, run
        if not paused:
            vid.cur_frame += 1
        else:
            pass

        #advance by appropriate number of frames (or reset to )
        if ('bNextF' in event) or ('bPrevF' in event) or ('Reset' in event):
            buttonDict = {
            'bPrevF' : -1, 'bPrevF5' : -5, 'bPrevF30' : -30,  
            'bNextF' : 1, 'bNextF5' : 5, 'bNextF30' : 30, 
            'bReset' : 0}
            numAdvance = buttonDict.get(event)
            back = vid.cur_frame
            if numAdvance == 0:
                vid.cur_frame = 0
            vid.cur_frame += numAdvance
            #if it ended and they are going back (must reset timeout)
            if timeout == None:
                timeout = 1000 // vid.fps
            if vid.cur_frame >= vid.tot_frames:
                print('This is the end of the video. Please click \"Next Video\" to advance')
            slider_elem.update(vid.cur_frame)
            window['sleft'].update(f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')
       
       # Jump to frame number or time stamp 
       # -- other formats? jump to minute or something? 
       # -- prefer diff boxes for frame# vs time stamp/is frame# useful or just confusing 
        elif event == '-CONFIRM-INPUT-':
            timef = values['-INPUT-TIME-']
            hold = vid.cur_frame
            # if frame number format, jump to indicated frame 
            try:
                vid.cur_frame = int(timef)
                back = hold
                if vid.cur_frame > vid.tot_frames:
                    print('There are only', int(vid.tot_frames), 'frames in the video. Please select a frame within the video')
                    continue 
                slider_elem.update(vid.cur_frame)
                window['sleft'].update(f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')
            except:
                # if time stamp format, jump to indicated time stamp
                try: 
                    m, s = timef.split(':')
                    back = hold
                    vid.cur_frame = round(vid.fps * (int(m)*60 + int(s)))
                    slider_elem.update(vid.cur_frame)
                    window['sleft'].update(f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')
                # incorrect formatting
                except:
                    print('Time is not in valid form %M:%S or frame#')
    
        # Jump one step backwards (from jump or button press)
        # currently only saves one step back, but we can save more back spaces if wanted 
        elif event == '-back-':
            vid.cur_frame = back
            slider_elem.update(vid.cur_frame)
            window['sleft'].update(f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))} .{int((vid.cur_frame % vid.fps) * 3.3):02d}')

        elif event == '-RUN-PAUSE-':
            # cv2.waitKey(-1)
            paused = not paused # flip value of 'paused'
            window['-RUN-PAUSE-'].Update('Run' if paused else 'Pause')
        #"Next Video" button => update info on slide with next video 

        elif event == '-gtp-':      # get trim points based on current frame
                    print('Approximated trials\' start/end to video frames (min:sec)...')
                    if vid.trial_num == 0:
                        vid.trial_num = 12

                    if debug:
                        print('Generated arrays for trim points and labels')

                    t.trim_starts = list()
                    t.trim_ends = list()
                    t.trim_labels = list()
                    
                    for i in range(vid.trial_num):
                        start_frame = vid.cur_frame + 240*vid.fps*i
                        end_frame = start_frame + 120*vid.fps

                        if (start_frame > vid.tot_frames) or (end_frame > vid.tot_frames):
                            window[trial_list[i] + '_start'].update('Out of frames', text_color='black')
                            window[trial_list[i] + '_end'].update('Out of frames', text_color='black')
                            window[f'{trial_list[i]}_ADD_TRIMPTS'].update(disabled = True)
                            window[f'{trial_list[i]}_gstart'].update(disabled = True)
                            window[f'{trial_list[i]}_gend'].update(disabled = True)
                        else:
                            t.trim_starts.append(start_frame)
                            t.trim_ends.append(end_frame) 
                            window[trial_list[i] + '_start'].update(t.frame_to_time(start_frame, vid.fps), text_color='black')                      
                            window[trial_list[i] + '_end'].update(t.frame_to_time(end_frame, vid.fps), text_color='black')
                            window[f'{trial_list[i]}_ADD_TRIMPTS'].update(disabled = False)
                            window[f'{trial_list[i]}_gstart'].update(disabled = False)
                            window[f'{trial_list[i]}_gend'].update(disabled = False)
                            
                        t.trim_labels.append('clip_0')
                        window[trial_list[i] + '_label'].update(t.trim_labels[i])
                    # to track actual trial ends when modifying for clips
                    original_t_starts = t.trim_starts.copy()
                    original_t_ends = t.trim_ends.copy()

        elif event in [i+'_gstart' for i in trial_list] or event in [i+'_gend' for i in trial_list]:  # if goto button pressed, go to start frame of trim range
            if event.endswith('gstart'):
                vid.cur_frame = t.trim_starts[int(event.split('_')[0][1:])-1]
            elif event.endswith('gend'):
                vid.cur_frame = t.trim_ends[int(event.split('_')[0][1:])-1]
            slider_elem.update(vid.cur_frame)
            window['sleft'].update(
            f'{strftime("%M:%S", gmtime(vid.cur_frame / vid.fps))}.{int((vid.cur_frame % vid.fps) * 3.3)}') # TODO: use fx for this

            vid.frame = cv2.resize(vid.frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
            imgbytes = cv2.imencode('.ppm', vid.frame)[1].tobytes()
            image_elem.update(data=imgbytes)
        elif event == '-resize-':
            resize_factor = float(values['-input-resize-'])

        # TODO: do start > end frame check as below
        # if textbox start or end pressed, update time to current frame
        elif event in [f'{i}_start' for i in trial_list] or event in [f'{i}_end' for i in trial_list]:
            trial_idx = trial_list.index(event.split('_')[0])  # get index of trial for indexing arrays
            if debug:
                print(f'updating {event} to display frame {vid.cur_frame} as {t.frame_to_time(vid.cur_frame, vid.fps)}')
            try:
                if 'start' in event:
                    t.trim_starts[trial_idx] = vid.cur_frame
                elif 'end' in event:
                    t.trim_ends[trial_idx] = vid.cur_frame
                else:
                    raise Exception
                window[event].update(t.frame_to_time(vid.cur_frame, vid.fps))
            except Exception as err:
                print('could not parse event ({event}) from text click')

        

            # TODO: this can define a func to be used in -add_all- 
            #   taking in trial idx or event value as x, the rest is already in scope
            # TODO: prevent adding trimpoints that are start > end, or otherwise invalid / excluded?
            #   currently have to remove them manually
        # add trim points button pressed - per trial
        elif event in [f'{i}_ADD_TRIMPTS' for i in trial_list]:

            trial_idx = trial_list.index(event.split('_')[0])  # get index of trial for indexing arrays
            trim_start = t.trim_starts[trial_idx]
            trim_end = t.trim_ends[trial_idx]
            trim_label = values[trial_list[trial_idx] + '_label']

            # add info to export arrays
            t.add_trim_points(trial_idx, trim_start, trim_end, trim_label)
            
            # add to clip number, whether previous clip was custom or not
            num_clips = len(t.trim_pts_array[np.where(t.trim_pts_array[:,0]==trial_idx)])
            window[trial_list[trial_idx] + '_label'].update(f'clip_{num_clips}')

            t.get_printed_trimpoints_listbox(vid.fps)  # get trimpoints into list of strings for listbox
            window['-tpts_out-'].update(t.printout)   # print out current trimpoints added to 2nd notes textbox

            # set start tp box to end point of added trimpoint, and end tp to end of trial again - simplifies workflow
            window[f'{trial_list[trial_idx]}_start'].update(t.frame_to_time(t.trim_ends[trial_idx], vid.fps))
            t.trim_starts[trial_idx] = t.trim_ends[trial_idx]

            # TODO: simplify below... may be able to do a check per timeout outside of event, 
            #   but should only print debug msg once, or rely on a bool to prevent adding/saving when below is true
            # check is start trim point is ahead of trial end point
            if t.trim_starts[trial_idx] > original_t_ends[trial_idx]:
                if debug:
                    print('Trim points are both past trial end, or end time is earlier than start time')
                window[f'{trial_list[trial_idx]}_start'].update(text_color='red')
                window[f'{trial_list[trial_idx]}_end'].update(t.frame_to_time(t.trim_starts[trial_idx], vid.fps), text_color='red')
                t.trim_ends[trial_idx] = t.trim_starts[trial_idx]
            else:   # all good
                window[f'{trial_list[trial_idx]}_end'].update(t.frame_to_time(original_t_ends[trial_idx], vid.fps))
                t.trim_ends[trial_idx] = original_t_ends[trial_idx]
            print(f'Set current start point for trial {trial_idx+1} to end of clip added, and end point to original trial end time')
            
        elif event == '-add_all-':
            if t.trim_starts == []:
                popup('There are no trim points.\nPlease select \"Get trial timepoints\"')
            elif popup("This will add trim points for ALL trials based on their current times and label names \nWould you like to continue?", 'yesno'):
                for i in range(len(t.trim_starts)):
                    trial_idx = i  # get index of trial for indexing arrays
                    trim_start = t.trim_starts[trial_idx]
                    trim_end = t.trim_ends[trial_idx]
                    trim_label = values[trial_list[trial_idx] + '_label']
                    t.add_trim_points(trial_idx, trim_start, trim_end, trim_label)
                    num_clips = len(t.trim_pts_array[np.where(t.trim_pts_array[:,0]==trial_idx)])
                    window[trial_list[trial_idx] + '_label'].update(f'clip_{num_clips}')
                    # flip times as in add_trimpts
                    window[f'{trial_list[trial_idx]}_start'].update(t.frame_to_time(t.trim_ends[trial_idx], vid.fps))
                    t.trim_starts[trial_idx] = t.trim_ends[trial_idx]
                    if t.trim_starts[trial_idx] > original_t_ends[trial_idx]:
                        if debug:
                            print('Trim points are both past trial end, or end time is earlier than start time')
                        window[f'{trial_list[trial_idx]}_start'].update(text_color='red')
                        window[f'{trial_list[trial_idx]}_end'].update(t.frame_to_time(t.trim_starts[trial_idx], vid.fps), text_color='red')
                        t.trim_ends[trial_idx] = t.trim_starts[trial_idx]
                    else: # all goodprintout
                        window[f'{trial_list[trial_idx]}_end'].update(t.frame_to_time(original_t_ends[trial_idx], vid.fps))
                        t.trim_ends[trial_idx] = original_t_ends[trial_idx]
                t.get_printed_trimpoints_listbox(vid.fps)  # get trimpoints into list of strings for listbox
                window['-tpts_out-'].update(t.printout)
                
        # remove selected trim points
        elif event =='-remove_tp-':
            to_remove = listbox_elem.get()
            if to_remove == []:
                popup('You have not selected any trim points to remove. Please select trim points to remove and try again.')
            else:
                if popup("This will remove the trim point(s): \n" +  '\n'.join(to_remove) + "\n\nWould you like to continue?", 'yesno'):
                    if t.printout != []:
                        t.remove_selected_tp(listbox_elem)
                        t.get_printed_trimpoints_listbox(vid.fps)
                        listbox_elem.update(t.printout)
                    else:
                        print('Tried to remove trimpoints but no trimpoints have been added.')
        # clear all trim point boxes
        elif event == '-clr-':
            if popup("This will clear all trim point boxes AND added trim points. \nWould you like to continue?", 'yesno'):
                t.trim_pts_array = np.ndarray((vid.trial_num,3))
                t.trim_pts_array[:] = np.nan
                t.trim_pts_labels = np.ndarray((vid.trial_num,2), dtype='object')
                t.trim_starts = list()
                t.trim_ends = list()
                t.trim_labels = list()
                for i in range(0, vid.trial_num):
                    window[trial_list[i] + '_start'].update('')                    
                    window[trial_list[i] + '_end'].update('')
                    window[trial_list[i] + '_label'].update('')
                t.printout = list()
                listbox_elem.update(['No trim points added.'])
                if debug:
                    print('Cleared trim point times and export list.')
        
        # clear added trim points
        elif event == '-clear_tp-':
            if popup("This will clear all the added trim points and reset the trim point boxes. \nWould you like to continue?", 'yesno'):
                t.trim_pts_array = np.ndarray((vid.trial_num,3))
                t.trim_pts_array[:] = np.nan
                t.trim_pts_labels = np.ndarray((vid.trial_num,2), dtype='object')
                for i in range(vid.trial_num):
                            t.trim_labels.append('clip_0')
                            window[trial_list[i] + '_label'].update(t.trim_labels[i])
                t.printout = list()
                listbox_elem.update(['No trim points added.'])
                t.trim_labels = list()
                if debug:
                    print('Cleared added trim points.')

        # save click
        elif event == '-save-':
            if debug:
                print('Saving...')
            # TODO: 
            #   Organize by component exported, ie clips, summary, metadata
            out_name = window['-preview-'].get()
            label_base = out_name
            print('debug: label_base = ', label_base)
            print('old ', f'{values["-mouseID-"]}_{values["-sess-"]}')

            # remove nans from arrays if present
            if any([np.isnan(ix) for ix in t.trim_pts_array[:,0]]):
                nans_idx = np.where([np.isnan(ix) for ix in t.trim_pts_array[:,0]])[0]
                t.trim_pts_array = np.delete(t.trim_pts_array, nans_idx, axis=0)
                t.trim_pts_labels = np.delete(t.trim_pts_labels, nans_idx, axis=0)
            clip_labels_list = list()
            print(t.trim_pts_labels)
            for trial_ind, label in t.trim_pts_labels[:]:
                    clip_label = f'{label_base}_t{int(trial_ind)+1:02d}_{label}'
                    clip_labels_list.append(clip_label)
            clip_labels_array = np.array(clip_labels_list, dtype='object')

            # create clips output folder if not present
            abs_out_path = os.path.abspath(folder_out)
            if not os.path.isdir(abs_out_path):
                os.mkdir(abs_out_path)
            out_base = os.path.join(abs_out_path, out_name)

            # save numpy arrays
            np.save((trimpts_name:= f'{out_base}_times.npy'), t.trim_pts_array[:,1:])
            np.save((labels_name:= f'{out_base}_labels.npy'), clip_labels_array)

            # TODO: this could be a func to build csv and export using arrays/info
            trial_col = pd.Series(t.trim_pts_array[:,0]+1, name='Trial#', dtype=int)
            start_col = pd.Series(t.frame_to_time(t.trim_pts_array[:,1].tolist(), vid.fps), name='Start')
            end_col = pd.Series(t.frame_to_time(t.trim_pts_array[:,2].tolist(), vid.fps), name='End')
            # prepend tab char \t so excel doesn't mess up csv time formatting
            start_col_tabs = '\t' + start_col[:]
            end_col_tabs = '\t' + end_col[:]

            label_col = pd.Series(clip_labels_array, name='ClipName')
            out_csv = pd.concat([trial_col, label_col, start_col_tabs, end_col_tabs], axis=1)
            out_csv.to_csv(f'{out_base}.csv', index=True, index_label='npy_array_index', date_format='%M:%S')

            # metadata section
            # gather metadata for current video and combine into dict with filename as head of tree
            video_info = {
                    'filename': fileName,
                    'filepath': filePath,
                    'num_frames': vid.tot_frames,
                    'height:width':f'{vid.height}:{vid.width}',
                    'FPS': vid.fps,
                    'Duration (h:min:sec)': strftime("%H:%M:%S", gmtime(vid.tot_frames/vid.fps))
            }
            expt_info = {
                    'mouse': values["-mouseID-"],
                    'experiment': values["-pID-"],
                    'session': values["-sess-"],
                    'date': datetime.strptime(values["-date-"], "%y%m%d").strftime("%Y.%m.%d")
            }
            clips_details = {}
            for clip, start, end in zip(label_col, start_col, end_col):
                clips_details[clip] = f'{start} to {end}'            
            export_info = {
                    'times_path': trimpts_name,
                    'labels_path': labels_name,
                    'total_num_clips': len(clip_labels_array),
                    'user': values["-user-"],
                    'output_name': out_base,
                    'output_folder': abs_out_path,
                    'summary_path': None, # not sure if the summary csv will stay, or be used for this
                    'clips_details': clips_details,
                    'date_processed': datetime.today().strftime("%Y.%m.%d"),
            }
            video_metadata = {
                fileName: {
                        'video_info': video_info,
                        'experiment_info': expt_info,
                        'export_info': export_info,
                        'trimmed': False,   # when ffmpeg script trims, it will set this to true for that video
                }
            }

            # append to metadata if present, otherwise create
            meta_filepath = os.path.join(abs_out_path, 'metadata.yaml')
            if os.path.isfile(meta_filepath):
                # load and append metadata
                out_meta = load_metadata(meta_filepath)
                out_meta.update(video_metadata) # adds new fileName entry
                print(f'\nAppending new metadata to "{meta_filepath}"\n', yaml.dump(video_metadata))
            else:
                # create new output
                out_meta = video_metadata
                print(f'\nCreated new metadata file: {meta_filepath}\n', yaml.dump(out_meta))
            # save to yaml file
            save_metadata(out_meta, meta_filepath)

        # set current frame and read from stream
        vidFile.stream.set(cv2.CAP_PROP_POS_FRAMES, vid.cur_frame)    
        vid.ret, vid.frame = vidFile.stream.read()
    
        if vid.ret and timeout == None:
            timeout = 1000//vid.fps
        if not vid.ret:
            print('no frames')
            timeout=None
#             cur_frame = int(values['-SLIDER-'])
#             vidFile.stream.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)  # set new frame
#             slider_elem.update(cur_frame) 
#             if ret:
#                 print("AYO")
#             timeout = 1000//fps
            continue 
            
        # retrieve frame data from video at position = cur_frame
        #frame = imutils.resize(frame, width = 1000)
        window['sright'].update(f'{strftime("%M:%S", gmtime(vid.tot_frames / vid.fps))}')

        vid.frame = cv2.resize(vid.frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
        imgbytes = cv2.imencode('.ppm', vid.frame)[1].tobytes()  # can also use png. ppm found to be more efficient
        image_elem.update(data=imgbytes)

    window.close()  # close gui
    if vidFile:
        vidFile.stop()  # if video still open, close
        
if __name__=='__main__':
    main()