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
    update fileName, filePath, Slider(num_frames), info column, checkboxes (default=next(reward), text_color=next(r_col))
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


    # shows trim-points extracted from video, reward info for each trial, and excluded trials after pressing the GTP button
    # Also shows got to buttons, which can be used to locate a the start of a particular trial, and select frame buttons, which can be used to update the start time of a trial.
    # and exit buttons
    trim_col = [ 
                [
                #sg.Text('', size=(6, 1), pad=(1,0)),
                sg.B('Get trial timepoints', k='-gtp-', expand_x=False, size=(20,1), font='Arial 12', pad=(0, (0,5))),
                sg.B('Add all', font='Arial 12', k='-add_all-', auto_size_button=True, pad=(12, (0,5))),
                sg.B('Clear all', font='Arial 12', k='-clr-', auto_size_button=True, pad=(0, (0,5)))],
                [
                    sg.T('Trial', font='Arial 10', size=(3, 1), pad=(0,0)), 
                    sg.Combo([i for i in range(1,13)], key = 'drop_box', pad = (0,20,0,0), enable_events=True),
                    sg.T('Start', font='Arial 10', size=(10, 1), justification='center', pad=(0,0,0,10)),
                    sg.T('End', font='Arial 10', size=(10,1), justification='left', pad=(0,0,0,0)),
                    sg.T('Exclude?', font='Arial 10', 
                         size=(10,1), justification='right', pad=((0,0),(0,0))),
                    # sg.T('Exclude?', font='Arial 8', size=(7, 1), justification='left', expand_x=False),
                ]
            ]

    #normally -> default=next(reward), text_color=next(r_col)
    #iterator should still work in for loop though...
    global trial_list
    trial_list = list()
    for i in range(1,13):
        trim_col += [sg.Input('', k=f't{i}_label', expand_x=True, font='Arial 8', size=(5, 1), pad=(0, 1)),
                    # sg.B('Select Frame', font='Arial 7', k=f'gst{i}', size=(5,1)),
                    sg.B('GOTO', font='Arial 7', k=f't{i}_gstart', size=(5,1)),
                    sg.T('', k=f't{i}_start', font='Arial 8', relief='groove', background_color='white', size=(8, 1), pad=(0, 1), enable_events=True),
                    sg.T('', k=f't{i}_end', font='Arial 8', relief='groove', background_color='white', size=(8, 1), pad=(0, 1), enable_events=True),
                    sg.B('GOTO', font='Arial 7', k=f't{i}_gend', size=(5,1), pad=((5,10),1)),
                    sg.Checkbox("", k=f't{i}_exclude'),
                    # sg.CB('', key=f'-ex{i}-', default=False, disabled=True, enable_events=True)
                    ],
        trial_list.append(f't{i}')

    # top right of GUI, has animal and session info from behavior file
    # incorporates trim column layout for ease when constructing window
    info_column = [[sg.Column(trim_col, pad=(0, 0), vertical_alignment='bottom', size = (350,375))],
                   [sg.B('Save Trial', k= 'save_edited_trial'), sg.B('Clear Trial', k = 'clear_edited_trial')],
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
    window = sg.Window('INTEGRATE Basic Trimmer', layout, finalize=True, enable_close_attempted_event=True,
resizable=True)
    return window


# def fromFileName(fileName):
#     """
#     takes file name and returns appropriate session info
#     """
#     fn_split = fileName.split(sep='_')  # split filename into info components, extracted in lines below
#     mouse = fn_split[1]#[:-4]
#     #curr_m = video_db.loc[video_db['Mouse_ID'] == mouse] # get mouse ID
#     #curr_m_idx = [i for i,x in enumerate(video_db['Mouse_ID'].values) if x == mouse][0] # locate the index for each animal to find specific video files
#     phase = fn_split[0] 
#     day = fn_split[5]
#     date = datetime.strptime(fn_split[6], "%Y%m%d").strftime('%m/%d/20%y')
#     e_ID = fn_split[2]                        # get expt ID
#     #vidlist = curr_m['video_list'].values[0]  # get list of videos for animal (from video renaming script)
    
#     curr_sess_idx = (event_range['mouse_id'] == mouse.replace('-', '')) & (event_range['eday'] == day)
#     return mouse, phase, day, date, e_ID, curr_sess_idx


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


def frame_to_time(frame, fps, time_format="%M:%S"):
    """Converts frametime to time format based on FPS, such as for the trimpoint textboxes and slider text


    Parameters
    ----------
    frame : Any or list[Any]
        Frame number from video stream. Can be list of frame numbers.
    fps : int
        Frames per second rate from video stream.
    time_format : str, optional
        Time format to convert frame to, by default "%M:%S". eg 04:14 for 4 min, 14 sec

    Returns
    -------
    list[str] or str
        Formatted string for given frame. If frame is a list, returns list of formatted strings.
    """
    if isinstance(frame, list):
        return [strftime(time_format, gmtime(round(f / fps))) for f in frame]
    else:
        return strftime(time_format, gmtime(round(frame / fps)))


# TODO: this get trim points func, maybe
# def get_trim_points(window, trial_num=12):
#     """
#     create outputs and fill with trial start and end points per trial
#     generates:
#         - np arrays per trial with 2 columns, trial# and tuple(start,end)
#         - subsequent trim points will be added to respective array
#         - concatenated before export to yield a .npy with variable # of trim points per trial in one long 2-column array of ints, tuples 
#     """


def get_printed_trimpoints_listbox(framestend, t_labels, fps) -> list:
    """Creates list for current trim points to be exported per trial.
    This version makes a list of strings for selectable listbox elements

    Parameters
    ----------
    framestend : np.array
        Array with 2 columns for trim start points and trim end points **by frame number**.
    t_labels : np.array
        Array with 1 column for clip labels.
    fps : int
        Frames per second rate of video.

    Returns
    -------
    list
        List of strings describing clip label and corresponding trim points for item in array of trimpoints.
    """
    
    printout = list()
    for index, label in enumerate(t_labels):
        try:
            printout.append(f'{label}: {frame_to_time(framestend[index][0], fps)}-{frame_to_time(framestend[index][1], fps)}')
        except ValueError as err:
            # printout.append([f'trial {t[0]}: {t[1]}-{t[2]}'])
            continue
    return printout


def add_trim_points(trial_idx:list, trim_pts_array:np.ndarray, trim_pts_labels:np.ndarray, 
                    trim_start, trim_end, trim_label, debug=False):
    """Adds trimpoints for given trial index to trim points array created from get trim points button.
    Currently for selectable listbox version of output.

    Parameters
    ----------
    trial_idx : list
        List of trial indices
    trim_pts_array : np.ndarray
        Array with 3 columns for trial indices, trim start points and trim end points.
    trim_pts_labels : np.ndarray
        Array with 2 columns for trial indices and clip labels for above trim points.
    trim_start :
        Trim start frame.
    trim_end :
        Trim end frame.
    trim_label : str
        Label for clip.
    debug : bool, optional
        Debug printing, by default False

    Returns
    -------
    Input trim point arrays with modified/appended rows. 
        trim_pts_array: np.ndarray
        trim_pts_labels: np.ndarray
        
    """
    if any([np.isnan(ix) for ix in trim_pts_array[:,0]]):    # if any uninitialized rows
        nan_idx = np.where([np.isnan(ix) for ix in trim_pts_array[:,0]])[0][0]  # get first index where above is true
        trim_pts_array[nan_idx] = np.array([[trial_idx, trim_start, trim_end]])     # replace nan values if any
        trim_pts_labels[nan_idx] = np.array([[trial_idx, trim_label]]) 
        if debug:
            print(f'added trimpoint frames {trim_pts_array[nan_idx][1]:.0f} : {trim_pts_array[nan_idx][2]:.0f} to trial {trial_idx+1}, ',
                  f'clip labeled "{trim_pts_labels[nan_idx][1]}"')
    else:
        trim_pts_array = np.append(trim_pts_array, np.array([[trial_idx, trim_start, trim_end]]), axis=0)
        trim_pts_labels = np.append(trim_pts_labels, np.array([[trial_idx, trim_label]]), axis=0)
        if debug:
            print(f'added trimpoint frames {trim_pts_array[-1][1]:.0f} : {trim_pts_array[-1][2]:.0f} to trial {trial_idx+1}, ', 
                  f'clip labeled "{trim_pts_labels[-1,1]}"')
    sort_idx = trim_pts_array[:,0].argsort()
    trim_pts_array = trim_pts_array[sort_idx]
    trim_pts_labels = trim_pts_labels[sort_idx]
    if debug:
        print('trimpts_array\n', trim_pts_array)
        print('trimpts_labels\n', trim_pts_labels)
    return trim_pts_array, trim_pts_labels


def remove_selected_tp(listbox_element: sg.Listbox, printout: list, 
                       trimpt_array: np.ndarray, trimpt_array_labels: np.ndarray, debug=False):
    """Retrieves selected elements from listbox of trimpoints to add, removes them.

    Parameters
    ----------
    listbox_element : sg.Listbox
        Output in GUI that displays added trim points for clips.
    printout : list
        List of trim points being displayed.
    trimpt_array : np.ndarray
        Array with 3 columns for trial indices, trim start points and trim end points.
    trimpt_array_labels : np.ndarray
        Array with 2 columns for trial indices and clip labels for above trim points.
    debug : bool, optional
        Debug printing, by default False

    Returns
    -------
    trim_pts_array: np.ndarray
    trim_pts_labels: np.ndarray
        Returns input trim point arrays with modified/appended rows.
    """

    to_remove = listbox_element.get()
    if debug:
        print('to remove: ', ["\n"+i for i in to_remove])
    if to_remove is not []:
        for item in to_remove:
            # remove_idx = listbox_element.get_list_values().index(item)
            remove_idx = printout.index(item)
            trimpt_array = np.delete(trimpt_array, remove_idx, axis=0)
            trimpt_array_labels = np.delete(trimpt_array_labels, remove_idx, axis=0)
            printout.remove(item)   # remove and then update list
        if debug:
            print('trimpts_arrays\n', trimpt_array, trimpt_array_labels)
        return trimpt_array, trimpt_array_labels
    else:
        print('Tried to remove trimpoints but none were selected.')


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

def find_metadata (fileName): 
    '''
    Matches video to appropriate metadata line from metadata file 
    '''
    #should only open it once in the main text - change later 
    file_path = r'C:\Users\virgi\OneDrive\Lab\behavioral_metadata_mats\sampleMeds\CSV_Out\metadata.pkl'
    video_metadata = pd.read_pickle(file_path).iloc[104]
    print("video_metadata", video_metadata)
    return video_metadata

def get_trim_pts (cur_metadata, fps): 
    timestend = np.empty((0,2))
    framestend = np.empty((0,2))
    t_labels = list()
    index_list = list()
    for event_name, time_pairs in cur_metadata.iloc[9:].items():
        for index, time_pair in enumerate(time_pairs):
            if time_pair == None:
                continue
            start_frame = time_pair[0] * fps
            end_frame = time_pair[1] * fps
            index_list.append(index)
            t_labels.append(f't{index+1}_' + event_name) 
            # print([time_pair[0], time_pair[1]])
            timestend = np.vstack((timestend, [time_pair[0], time_pair[1]]))
            framestend = np.vstack((framestend, [start_frame, end_frame]))

    # sort the arrays by t_labels such that they are by trial 
    indices = np.argsort(index_list)
    t_labels = np.array(t_labels)[indices]
    timestend = np.array(timestend)[indices]
    framestend = np.array(framestend)[indices]
    print("framestend", framestend)
    print("t_labels", t_labels)
    print("timestend", timestend)
    return timestend, framestend, t_labels
        
# %% Main

def main(debug=False):  
    #sg.theme('Kayak')
    sg.theme('Dark Blue 3')
    
    #1 ---------------- build layout/initialize variable 
    window = buildWindow(debug=debug)
    cur_frame = 0
    vidFile = None
    num_frames = 0
    #height, width = 0, 0
    ret, frame = None, None
    fps = 0    
    buttonDict = {
        'bPrevF' : -1, 'bPrevF5' : -5, 'bPrevF30' : -30,  
        'bNextF' : 1, 'bNextF5' : 5, 'bNextF30' : 30, 
        'bReset' : 0}
    cur_metadata = None
    paused = True   # start video paused
    stop = False    # used to stop loop for some events
    # window refresh rate in ms. 
    #   None until vid loaded, then 1000/frames per sec rate to match video play
    timeout = None 
    trial_num = 0
    printout = list()   # will output list of trimpoints added to be saved
    back = 0 
    trim_pts_array = np.ndarray((trial_num,3))
    trim_pts_array[:] = np.nan
    trim_pts_labels = np.ndarray((trial_num,2), dtype='object') # can probably do |S10 and decode but this is easier
    # go through and bind info input boxes to focus in/out (click on, out of), for clearing default text
    #   this may go, depending on how useful it is / useful the defaults are
    inputs = ['-mouseID-', '-pID-', '-sess-', '-date-', '-user-']
    input_focusin = list()
    input_focusout = list()
    input_defaults = list(['']*len(inputs))
    t_starts = list()
    t_ends = list()
    t_labels = list()
    resize_factor = 1
    timestend = np.empty((0,2))
    framestend = np.empty((0,2))
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
            # if not popup("Do you really want to exit?\n(Have you saved your work?)", 'yesno'):
            #     continue
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
            cur_frame = 0

            #load next video 
            vidFile = FileVideoStream(filePath, queue_size=200)
            sleep(2.0)  # give time to load video
            vidFile.start()
            sleep(8.0)  # give time for thread to start up
            
            #reset variables with video information
            num_frames = vidFile.stream.get(cv2.CAP_PROP_FRAME_COUNT)   
            ret, frame = vidFile.stream.read()
            frame = imutils.resize(frame, width = 1000)
            height, width = vidFile.stream.get(cv2.CAP_PROP_FRAME_HEIGHT), vidFile.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            print('Height: ',  height , ' Width: ' , width)
            print('my screen size; ', sg.Window.get_screen_size())
            fps = round(vidFile.stream.get(cv2.CAP_PROP_FPS))
            #window['sright'].update(f'{strftime("%M:%S", gmtime(num_frames / fps))}')

            print('Video FPS: ', fps)
            timeout = 1000 // fps
            
            #reset slider, image 
            slider_elem.update(0, range = (0, num_frames))   
            frame = cv2.resize(frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
  
            imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
            image_elem.update(data=imgbytes)
            for input in inputs:
                input_focusin.append(f'{input}_InFocus')
                window[input].bind('<FocusIn>', '_InFocus')
                input_focusout.append(f'{input}_OutFocus')
                window[input].bind('<FocusOut>', '_OutFocus')
                dict.update({input: ''})
                        # MATCHING NEED TO FIGURE OUT 
            cur_metadata = find_metadata(fileName)

            # Get trim points from metadata, update printout box
            timestend, framestend, t_labels = get_trim_pts(cur_metadata, fps = fps)
            printout = get_printed_trimpoints_listbox(framestend, t_labels, fps)
            listbox_elem.update(printout)

        # if just started program, must click "Next Video"
        elif vidFile == None:
            print('Please choose a video folder')
            continue 

        if event == 'bFrames':  # debug=True dependent
            print('Current Frame: ', cur_frame)
            print('num frames: ', num_frames)
            print('timeout: ', timeout)
            
        # cur_frame runs ahead of the last loaded image, 
        #   so if previous frame was the end frame, can't move forward
        if cur_frame - 1 == num_frames and event == 'bNextF': 
            print('Video has ended. Please click \"Next Video\"')
            continue

        # if end frame within 30 frames, give warning
        elif cur_frame + 31 > num_frames and event == 'bNextF30':
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
        if int(values['-SLIDER-']) != cur_frame - 1:    
            cur_frame = int(values['-SLIDER-'])
            #vidFile.stream.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)  # set new frame
            slider_elem.update(cur_frame)                           # update scrollbar position and
            # TODO: there's gotta be a better way to line sleft and sright text with the scrollbar,
            #   currently just padding from above to bring them down a bit
            window['sleft'].update(                                 #   relevant surrounding values
                f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')
            if ret: #may not be needed   
                frame = cv2.resize(frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
                try:
                    imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
                except:
                    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
                    print('tried png')
            image_elem.update(data=imgbytes)  # update shown frame

        slider_elem.update(cur_frame)
        window['sleft'].update(f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')

        # if not paused, run
        if not paused:
            cur_frame += 1
        else:
            pass

        #advance by appropriate number of frames (or reset to )
        if ('bNextF' in event) or ('bPrevF' in event) or ('Reset' in event):
            numAdvance = buttonDict.get(event)
            back = cur_frame
            if numAdvance == 0:
                cur_frame = 0
            cur_frame += numAdvance
            #if it ended and they are going back (must reset timeout)
            if timeout == None:
                timeout = 1000 // fps
            if cur_frame >= num_frames:
                print('This is the end of the video. Please click \"Next Video\" to advance')
            slider_elem.update(cur_frame)
            window['sleft'].update(f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')
       
        if event == 'drop_box':
            row_num = 1
            for label, stend, fstend in zip(t_labels, timestend, framestend) :
                if label.startswith("t" + str(values['drop_box'])):
                    window[f't{row_num}_label'].update(label, text_color='black')
                    window[f't{row_num}_start'].update(frame_to_time(fstend[0], fps), text_color='black')
                    window[f't{row_num}_end'].update(frame_to_time(fstend[1], fps), text_color='black')
                    row_num += 1
                    print(label, stend, fstend)
            print(trim_pts_labels)
            selected_option = values['drop_box']
            print(f'You selected: {selected_option}')
                
       # Jump to frame number or time stamp 
       # -- other formats? jump to minute or something? 
       # -- prefer diff boxes for frame# vs time stamp/is frame# useful or just confusing 
        elif event == '-CONFIRM-INPUT-':
            timef = values['-INPUT-TIME-']
            hold = cur_frame
            # if frame number format, jump to indicated frame 
            try:
                cur_frame = int(timef)
                back = hold
                if cur_frame > num_frames:
                    print('There are only', int(num_frames), 'frames in the video. Please select a frame within the video')
                    continue 
                slider_elem.update(cur_frame)
                window['sleft'].update(f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')
            except:
                # if time stamp format, jump to indicated time stamp
                try: 
                    m, s = timef.split(':')
                    back = hold
                    cur_frame = round(fps * (int(m)*60 + int(s)))
                    slider_elem.update(cur_frame)
                    window['sleft'].update(f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')
                # incorrect formatting
                except:
                    print('Time is not in valid form %M:%S or frame#')
    
        # Jump one step backwards (from jump or button press)
        # currently only saves one step back, but we can save more back spaces if wanted 
        elif event == '-back-':
            cur_frame = back
            slider_elem.update(cur_frame)
            window['sleft'].update(f'{strftime("%M:%S", gmtime(cur_frame / fps))} .{int((cur_frame % fps) * 3.3):02d}')

        elif event == '-RUN-PAUSE-':
            # cv2.waitKey(-1)
            paused = not paused # flip value of 'paused'
            window['-RUN-PAUSE-'].Update('Run' if paused else 'Pause')
        #"Next Video" button => update info on slide with next video 

        elif event == '-gtp-':      # get trim points based on current frame
            # print('Approximated trials\' start/end to video frames (min:sec)...')
            if trial_num == 0:
                trial_num = 12

            if debug:
                print('Generated arrays for trim points and labels')                

            t_starts = list()
            t_ends = list()
            t_labels = list()
            
            for event_name, time_pairs in cur_metadata.iloc[9:].items():
                for index, time_pair in enumerate(time_pairs):
                    if time_pair == None:
                        continue
                    start_frame = time_pair[0] * fps
                    end_frame = time_pair[1] * fps
                    
                    t_starts.append(start_frame)
                    t_ends.append(end_frame)
                    t_labels.append(f't{index+1}_' + event_name) 
                    
            for t in range(trial_num):
                start_frame = cur_frame + 240*fps*t
                end_frame = start_frame + 120*fps

                if (start_frame > num_frames) or (end_frame > num_frames):
                    window[trial_list[t] + '_start'].update('Out of frames', text_color='black')
                    window[trial_list[t] + '_end'].update('Out of frames', text_color='black')
                    window[f'{trial_list[t]}_ADD_TRIMPTS'].update(disabled = True)
                    window[f'{trial_list[t]}_gstart'].update(disabled = True)
                    window[f'{trial_list[t]}_gend'].update(disabled = True)
                else:
                    t_starts.append(start_frame)
                    t_ends.append(end_frame) 
                    window[trial_list[t] + '_start'].update(frame_to_time(start_frame, fps), text_color='black')                      
                    window[trial_list[t] + '_end'].update(frame_to_time(end_frame, fps), text_color='black')
                    window[f'{trial_list[t]}_ADD_TRIMPTS'].update(disabled = False)
                    window[f'{trial_list[t]}_gstart'].update(disabled = False)
                    window[f'{trial_list[t]}_gend'].update(disabled = False)
                    
                t_labels.append('clip_0')
                window[trial_list[t] + '_label'].update(t_labels[t])
            # to track actual trial ends when modifying for clips
            original_t_starts = t_starts.copy()
            original_t_ends = t_ends.copy()

        elif event in [i+'_gstart' for i in trial_list] or event in [i+'_gend' for i in trial_list]:  # if goto button pressed, go to start frame of trim range
            if event.endswith('gstart'):
                cur_frame = t_starts[int(event.split('_')[0][1:])-1]
            elif event.endswith('gend'):
                cur_frame = t_ends[int(event.split('_')[0][1:])-1]
            slider_elem.update(cur_frame)
            window['sleft'].update(
            f'{strftime("%M:%S", gmtime(cur_frame / fps))}.{int((cur_frame % fps) * 3.3)}') # TODO: use fx for this

            frame = cv2.resize(frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
            imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
            image_elem.update(data=imgbytes)
        elif event == '-resize-':
            resize_factor = float(values['-input-resize-'])

        # TODO: do start > end frame check as below
        # if textbox start or end pressed, update time to current frame
        elif event in [f'{i}_start' for i in trial_list] or event in [f'{i}_end' for i in trial_list]:
            trial_idx = trial_list.index(event.split('_')[0])  # get index of trial for indexing arrays
            if debug:
                print(f'updating {event} to display frame {cur_frame} as {frame_to_time(cur_frame, fps)}')
            try:
                if 'start' in event:
                    t_starts[trial_idx] = cur_frame
                elif 'end' in event:
                    t_ends[trial_idx] = cur_frame
                else:
                    raise Exception
                window[event].update(frame_to_time(cur_frame, fps))
            except Exception as err:
                print('could not parse event ({event}) from text click')

        

            # TODO: this can define a func to be used in -add_all- 
            #   taking in trial idx or event value as x, the rest is already in scope
            # TODO: prevent adding trimpoints that are start > end, or otherwise invalid / excluded?
            #   currently have to remove them manually
        # add trim points button pressed - per trial
        elif event in [f'{i}_ADD_TRIMPTS' for i in trial_list]:

            trial_idx = trial_list.index(event.split('_')[0])  # get index of trial for indexing arrays
            trim_start = t_starts[trial_idx]
            trim_end = t_ends[trial_idx]
            trim_label = values[trial_list[trial_idx] + '_label']

            # add info to export arrays
            trim_pts_array, trim_pts_labels = add_trim_points(trial_idx, trim_pts_array, trim_pts_labels, trim_start, trim_end, trim_label, debug)
            
            # add to clip number, whether previous clip was custom or not
            num_clips = len(trim_pts_array[np.where(trim_pts_array[:,0]==trial_idx)])
            window[trial_list[trial_idx] + '_label'].update(f'clip_{num_clips}')

            printout = get_printed_trimpoints_listbox(trim_pts_array, trim_pts_labels, fps)  # get trimpoints into list of strings for listbox
            window['-tpts_out-'].update(printout)   # print out current trimpoints added to 2nd notes textbox

            # set start tp box to end point of added trimpoint, and end tp to end of trial again - simplifies workflow
            window[f'{trial_list[trial_idx]}_start'].update(frame_to_time(t_ends[trial_idx], fps))
            t_starts[trial_idx] = t_ends[trial_idx]

            # TODO: simplify below... may be able to do a check per timeout outside of event, 
            #   but should only print debug msg once, or rely on a bool to prevent adding/saving when below is true
            # check is start trim point is ahead of trial end point
            if t_starts[trial_idx] > original_t_ends[trial_idx]:
                if debug:
                    print('Trim points are both past trial end, or end time is earlier than start time')
                window[f'{trial_list[trial_idx]}_start'].update(text_color='red')
                window[f'{trial_list[trial_idx]}_end'].update(frame_to_time(t_starts[trial_idx], fps), text_color='red')
                t_ends[trial_idx] = t_starts[trial_idx]
            else:   # all good
                window[f'{trial_list[trial_idx]}_end'].update(frame_to_time(original_t_ends[trial_idx], fps))
                t_ends[trial_idx] = original_t_ends[trial_idx]
            print(f'Set current start point for trial {trial_idx+1} to end of clip added, and end point to original trial end time')
            
        elif event == '-add_all-':
            if t_starts == []:
                popup('There are no trim points.\nPlease select \"Get trial timepoints\"')
            elif popup("This will add trim points for ALL trials based on their current times and label names \nWould you like to continue?", 'yesno'):
                for i in range(len(t_starts)):
                    trial_idx = i  # get index of trial for indexing arrays
                    trim_start = t_starts[trial_idx]
                    trim_end = t_ends[trial_idx]
                    trim_label = values[trial_list[trial_idx] + '_label']
                    trim_pts_array, trim_pts_labels = add_trim_points(trial_idx, trim_pts_array, trim_pts_labels, trim_start, trim_end, trim_label, debug)
                    num_clips = len(trim_pts_array[np.where(trim_pts_array[:,0]==trial_idx)])
                    window[trial_list[trial_idx] + '_label'].update(f'clip_{num_clips}')
                    # flip times as in add_trimpts
                    window[f'{trial_list[trial_idx]}_start'].update(frame_to_time(t_ends[trial_idx], fps))
                    t_starts[trial_idx] = t_ends[trial_idx]
                    if t_starts[trial_idx] > original_t_ends[trial_idx]:
                        if debug:
                            print('Trim points are both past trial end, or end time is earlier than start time')
                        window[f'{trial_list[trial_idx]}_start'].update(text_color='red')
                        window[f'{trial_list[trial_idx]}_end'].update(frame_to_time(t_starts[trial_idx], fps), text_color='red')
                        t_ends[trial_idx] = t_starts[trial_idx]
                    else: # all goodprintout
                        window[f'{trial_list[trial_idx]}_end'].update(frame_to_time(original_t_ends[trial_idx], fps))
                        t_ends[trial_idx] = original_t_ends[trial_idx]
                printout = get_printed_trimpoints_listbox(trim_pts_array, trim_pts_labels, fps)  # get trimpoints into list of strings for listbox
                window['-tpts_out-'].update(printout)
                
        # remove selected trim points
        elif event =='-remove_tp-':
            to_remove = listbox_elem.get()
            if to_remove == []:
                popup('You have not selected any trim points to remove. Please select trim points to remove and try again.')
            else:
                if popup("This will remove the trim point(s): \n" +  '\n'.join(to_remove) + "\n\nWould you like to continue?", 'yesno'):
                    if printout != []:
                        trim_pts_array, trim_pts_labels = remove_selected_tp(listbox_elem, printout, trim_pts_array, trim_pts_labels, debug)
                        printout = get_printed_trimpoints_listbox(trim_pts_array, trim_pts_labels, fps)
                        listbox_elem.update(printout)
                    else:
                        print('Tried to remove trimpoints but no trimpoints have been added.')
        # clear all trim point boxes
        elif event == '-clr-':
            if popup("This will clear all trim point boxes AND added trim points. \nWould you like to continue?", 'yesno'):
                trim_pts_array = np.ndarray((trial_num,3))
                trim_pts_array[:] = np.nan
                trim_pts_labels = np.ndarray((trial_num,2), dtype='object')
                t_starts = list()
                t_ends = list()
                t_labels = list()
                for t in range(0, trial_num):
                    window[trial_list[t] + '_start'].update('')                    
                    window[trial_list[t] + '_end'].update('')
                    window[trial_list[t] + '_label'].update('')
                printout = list()
                listbox_elem.update(['No trim points added.'])
                if debug:
                    print('Cleared trim point times and export list.')
        
        # clear added trim points
        elif event == '-clear_tp-':
            if popup("This will clear all the added trim points and reset the trim point boxes. \nWould you like to continue?", 'yesno'):
                trim_pts_array = np.ndarray((trial_num,3))
                trim_pts_array[:] = np.nan
                trim_pts_labels = np.ndarray((trial_num,2), dtype='object')
                for t in range(trial_num):
                            t_labels.append('clip_0')
                            window[trial_list[t] + '_label'].update(t_labels[t])
                printout = list()
                listbox_elem.update(['No trim points added.'])
                t_labels = list()
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
            if any([np.isnan(ix) for ix in trim_pts_array[:,0]]):
                nans_idx = np.where([np.isnan(ix) for ix in trim_pts_array[:,0]])[0]
                trim_pts_array = np.delete(trim_pts_array, nans_idx, axis=0)
                trim_pts_labels = np.delete(trim_pts_labels, nans_idx, axis=0)
            clip_labels_list = list()
            for trial_ind, label in trim_pts_labels[:]:
                    clip_label = f'{label_base}_t{int(trial_ind)+1:02d}_{label}'
                    clip_labels_list.append(clip_label)
            clip_labels_array = np.array(clip_labels_list, dtype='object')

            # create clips output folder if not present
            abs_out_path = os.path.abspath(folder_out)
            if not os.path.isdir(abs_out_path):
                os.mkdir(abs_out_path)
            out_base = os.path.join(abs_out_path, out_name)

            # save numpy arrays
            np.save((trimpts_name:= f'{out_base}_times.npy'), trim_pts_array[:,1:])
            np.save((labels_name:= f'{out_base}_labels.npy'), clip_labels_array)

            # TODO: this could be a func to build csv and export using arrays/info
            trial_col = pd.Series(trim_pts_array[:,0]+1, name='Trial#', dtype=int)
            start_col = pd.Series(frame_to_time(trim_pts_array[:,1].tolist(), fps), name='Start')
            end_col = pd.Series(frame_to_time(trim_pts_array[:,2].tolist(), fps), name='End')
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
                    'num_frames': num_frames,
                    'height:width':f'{height}:{width}',
                    'FPS': fps,
                    'Duration (h:min:sec)': strftime("%H:%M:%S", gmtime(num_frames/fps))
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
        vidFile.stream.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)    
        ret, frame = vidFile.stream.read()
    
        if ret and timeout == None:
            timeout = 1000//fps
        if not ret:
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
        window['sright'].update(f'{strftime("%M:%S", gmtime(num_frames / fps))}')

        frame = cv2.resize(frame, None, fx= resize_factor, fy= resize_factor, interpolation= cv2.INTER_LINEAR)
        imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()  # can also use png. ppm found to be more efficient
        image_elem.update(data=imgbytes)

    window.close()  # close gui
    if vidFile:
        vidFile.stop()  # if video still open, close
        
if __name__=='__main__':
    main()
# %%
