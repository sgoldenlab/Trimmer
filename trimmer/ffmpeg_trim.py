import os
import ffmpeg as ff
import numpy as np
import yaml
from asynccpu import ProcessTaskPoolExecutor
# from asyncffmpeg import FFmpegCoroutineFactory, StreamSpec
import asyncio
import PySimpleGUI as sg
import sys
import pdb

# %% Functions

def load_metadata(filepath, debug=False):
    """
    Checks if metadata exists and loads it to dictionary
    """
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            yaml_dict = yaml.load(file, yaml.BaseLoader)
        return yaml_dict
    else:
        if debug:
            print(f'Could not find {os.path.basename(filepath)}.')
        return None


def ffload(v_in):
    probe_res = ff.probe(v_in)
    v_duration = probe_res.get("format", {}).get("duration",None)
    print(v_duration)

    v_in_stream = ff.input(v_in)
    return v_in_stream


def get_clip_name(label, out_base_path, replace=False):
    clip_out_name = f'{label}.mp4' #'_'.join(['clip',orig_vname,'['+e_range+']', t_num])
    clip_out_path = os.path.join(out_base_path, clip_out_name)
    if os.path.isfile(clip_out_path):
            print(f'clip already exists: {os.path.basename(clip_out_path)}')
            if replace:
                os.remove(clip_out_path)
            else:
                print(f'skipping clip {clip_out_path}')
                return None
    return clip_out_path


def gui_prompt():
    """Simple gui prompt to request file selection for ffmpeg trimming parameters.

    Returns
    -------
    folder: str or None
        Path to metadata.yml or None if none selected before exiting.
    """
    left_col = [[sg.Text('Path to metadata.yaml folder:'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()]]
    layout = [[sg.Column(left_col, element_justification='c')] ]   
    window = sg.Window('Select folder containing metadata.yml to start trimming.', layout,resizable=True)
    folder = None
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == '-FOLDER-':
            window['-FOLDER-'].update(text_color='black')
            if 'metadata.yaml' not in os.listdir(values['-FOLDER-']):
                window['-FOLDER-'].update(text_color='red')
                window['-FOLDER-'].update('No metadata.yaml found. Please try again.')
                continue
            folder = values['-FOLDER-'] 
            break
    window.close()
    return folder


# %% Async functions
async def trim_process(task_id, v_in, t_start, t_end):
    """
    Description
    """
    pts = 'PTS-STARTPTS'
    result = (
        v_in
        .trim(start_frame=t_start, end_frame=t_end)
        .setpts(pts)
        .output(task_id, crf=18)
        .run()
    )
    return f"task_id: {task_id}, result: {result}"


# Non-async version
# def fftrim(v_in, v_out, trim_st, trim_end, replace = True, debug=False):
#     if os.path.isfile(v_out):
#         print(f'clip already exists: {os.path.basename(v_out)}')
#         if replace:
#             os.remove(v_out)
#         else:
#             print(f'skipping clip {v_out}')
#             return None
#     pts = 'PTS-STARTPTS'
#     (
#         v_in
#         .trim(start_frame=trim_st, end_frame=trim_end)
#         .setpts(pts)
#         .output(v_out)
#         .run()
#     )


# %% Main function

async def main(video, metadata, replace=False, debug=True):
    # for video in to_trim_list:
    # TODO: input() or in gui to ask whether to down sample, 
    #   diff fx or modified trim_process to include downsampling with ff.filter I think
    # video_info, _, export_info, isTrimmed = metadata[video].values() 
    _, export_info, isTrimmed, video_info = metadata[video].values() 

    assert isTrimmed in bools, f'video appears to have been trimmed already: {video}'
    # video info
    vid_path = video_info['filepath']
    num_frames = video_info['num_frames']
    fps = video_info['FPS']
    # height width?
    assert os.path.isfile(vid_path), f'no video found: {vid_path}'
    print(f'Trimming video: "{os.path.abspath(vid_path)}"')

    # export info
    times_path = export_info['times_path']
    labels_path = export_info['labels_path']
    num_clips = export_info['total_num_clips']
    out_name = export_info['output_name']

    assert os.path.isfile(times_path), f'no trim points array found: {times_path}'
    assert os.path.isfile(labels_path), f'no clip labels array found: {labels_path}'
    trim_points = np.load(times_path, allow_pickle=True)
    clip_labels = export_info['clips_details'] # np.load(labels_path, allow_pickle=True)
    
    
    
    # TODO: may change this to clips_excluded or something, if excluded checkboxes actually get used
    #   Currently excluded cboxes aren't needed since trimpoints need to be manually added.
    # keys_to_remove = [k for k, v in trim_points.items() if k not in ch_ranges]
    # for r in keys_to_remove:
    #     trim_points.pop(r,None) # remove unwanted event ranges

    v_in = ffload(vid_path)  # create input stream for ffmpeg
    clip_paths = [(get_clip_name(clip, out_base, replace=replace)) for clip in clip_labels]

    with ProcessTaskPoolExecutor(max_workers=3, cancel_tasks_when_shutdown=True) as executor:
        awaitables = {executor.create_process_task(trim_process, z, v_in, x, y,) 
                        for x,y,z in zip(trim_points[:,0], trim_points[:,1], clip_paths) if z is not None}
        results = await asyncio.gather(*awaitables)
    metadata[video].update({'trimmed': True})   # video now labeled as trimmed 
    print(f'...finished trimming clips for video "{os.path.abspath(video)}"')

if __name__ == '__main__':
    # %% Set up 
    debug = False
    # Get source folders for videos and metadata file, if present
    # meta_fold = r'./'
    
    # meta_fold = r'./test_folder/clips'
    # meta_filepath = os.path.join(meta_fold, 'metadata.yaml')
    meta_fold = gui_prompt()
    if not meta_fold:
        print('No folder selected. Closing...')
        quit()
    meta_filepath = os.path.join(meta_fold, 'metadata.yaml')

    # vids_fold = rf'../'   # changed this to get the paths from the metadata
    
    
    # check for metadata
    bools = ['false', 'False', False]   # potential false values
    if os.path.isfile(meta_filepath):
        metadata = load_metadata(meta_filepath) # load metadata yaml file
        videos = list(metadata.keys())

        # grab videos' filepaths to trim unless they've been trimmed already, ie metadata.videoname.trimmed == True
        to_trim_list = [vid for vid in videos 
                        if metadata[vid]['trimmed'] in bools]
        if debug:
            print('Videos to trim:\n', to_trim_list)
            exit()

    else:
        print(f'No metadata.yaml file found in {meta_fold}, please check path or use trimmer before executing script')

    out_base = meta_fold # unless changed, output is same as output from trimmer 
    # ch_ranges = ['post_door_close', 'pre_lever_in']#, 'pre_lev_press'] # not using behavioral data in this version

    # %% Main loop
    for video in to_trim_list:
        asyncio.run(main(video, metadata, replace=False, debug=False))
    print('\nFinished all trimming.')