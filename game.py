"""
Library for retrieving basektball player-tracking and play-by-play data.
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import warnings
import json
import time
import urllib.request
import py7zr
import pandas as pd
import shutil
import sys
import subprocess
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc, Polygon
import numpy as np
import seaborn as sns
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d

# Initialize project
if not os.path.exists('temp'):
    os.makedirs('temp', exist_ok=True)
datalink = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/01.13.2016.GSW.at.DEN.7z"


class Game(object):
    """
    Class for basketball game.
    Contains play by play and player tracking data and methods for
    anaylsis and plotting.
    """

    def __init__(self, date, team1, team2):
        """
        Args:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name

        Attributes:
            date (str): 'MM.DD.YYYY', date of game
            team1 (str): 'XXX', abbreviation of team1 in data
                tracking file name
            team2 (str): 'XXX', abbreviation of team2 in data
                tracking file name
            tracking_id (str): id to access player tracking data
                Due to the way the SportVU data is stored, game_id is
                complicated: 'MM.DD.YYYY.AWAYTEAM.at.HOMETEAM'
                For Example: 01.13.2016.GSW.at.DEN
            tracking_data (dict): Dictionary of unstructured tracking
                data scraped from github.
            game_id (str): ID for game.  Lukcily, SportVU and play by
                play use the same game ID
            pbp (pd.DataFrame): Play by play data.  33 columns per pbp
                instance.
            moments (pd.DataFrame): DataFrame of player tracking data.
                Each entry is a single snap-shot of where the players
                are at a given time on the court.
                Columns: ['quarter', 'universe_time', 'quarter_time',
                'shot_clock', 'positions', 'game_time'].
                moments['positions'] contains a list of where each player
                and the ball are located.
            player_ids (dict): dictionary of {player: player_id} for
                all players in game.
            away_id (int): ID of away team
            home_id (int): ID of home team
            team_colors (dict): dictionary of colors for each team and
                ball. Used for ploting.
            home_team (str): 'XXX', abbreviation of home team
            away_team (str): 'XXX', abbreviation of away team
        """
        self.date = date
        self.team1 = team1
        self.team2 = team2
        self.flip_direction = False
        self.tracking_id = ('{self.date}.{self.team2}.at.{self.team1}'
                            .format(self=self))
        self.tracking_data = None
        self.game_id = None
        self.pbp = None
        self.moments = None
        self.player_ids = None
        self._get_tracking_data()
        self._get_playbyplay_data()
        self._format_tracking_data()
        self._get_player_ids()
        self._get_player_jerseys()
        self.away_id = self.tracking_data['events'][0]['visitor']['teamid']
        self.home_id = self.tracking_data['events'][0]['home']['teamid']
        self.team_colors = {-1: "orange",
                            self.away_id: "blue",
                            self.home_id: "red"}
        self.home_team = (self.tracking_data['events'][0]['home']
                          ['abbreviation'])
        self.away_team = (self.tracking_data['events'][0]['visitor']
                          ['abbreviation'])
        self.flip_direction = False
        self._determine_direction()
        print('All data is loaded')

    def _get_tracking_data(self):
        """
        Helper function for retrieving tracking data
        Tracking Data is provided by NBA.com,
        hosted at: https://www.github.com/neilmj
        Update it is hosted now on https://github.com/sealneaward/nba-movement-data/tree/master
        """
        # Retrive and extract Data into /temp folder

        # Download the 7z file using urllib (cross-platform)
        print(f"Downloading data from {datalink}...")
        urllib.request.urlretrieve(datalink, "temp/game.7z")
        print("Download complete. Extracting...")
        
        # Extract using py7zr (pure Python implementation)
        with py7zr.SevenZipFile("temp/game.7z", mode='r') as archive:
            archive.extractall(path="temp")
        print("Extraction complete.")
        
        # Wait a moment for file handles to be released (Windows issue)
        time.sleep(0.5)
        
        # Try to remove the 7z file with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.remove("temp/game.7z")
                break
            except (PermissionError, FileNotFoundError):
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print("Warning: Could not delete temp/game.7z. You may need to delete it manually.")
        
        # Extract game ID from extracted file name.
        for file in os.listdir('./temp'):
            if os.path.splitext(file)[1] == '.json':
                self.game_id = file[:-5]

        # Load tracking data and remove json file
        with open('temp/{self.game_id}.json'.format(self=self)) as data_file:
            self.tracking_data = json.load(data_file)  # Load this json
        os.remove('./temp/{self.game_id}.json'.format(self=self))
        return self

    def _get_playbyplay_data(self):
        """
        Helper function for retrieving play-by-play data.
        Play-by-play data is obtained via API call to NBA.com
        This service is likely to go down at any moment and ruin this
        whole project.
        """
        # Download play-by-play data from GitHub
        pbp_link = "https://raw.githubusercontent.com/sealneaward/nba-movement-data/master/data/events/{self.game_id}.csv".format(self=self)
        pbp_filename = "temp/{self.game_id}_events.csv".format(self=self)
        
        print(f"Downloading play-by-play data from {pbp_link}...")
        urllib.request.urlretrieve(pbp_link, pbp_filename)
        print("Play-by-play download complete.")
        
        # load play by play into pandas DataFrame
        self.pbp = pd.read_csv(pbp_filename)
        os.remove(pbp_filename)

        # Get time in quarter reamining to cross-reference tracking data
        self.pbp['Qmin'] = (self.pbp['PCTIMESTRING'].str
                            .split(':', expand=True)[0])
        self.pbp['Qsec'] = (self.pbp['PCTIMESTRING'].str
                            .split(':', expand=True)[1])
        self.pbp['Qtime'] = (self.pbp['Qmin'].astype(int)*60 +
                             self.pbp['Qsec'].astype(int))
        self.pbp['game_time'] = ((self.pbp['PERIOD'] - 1) * 720 +
                                 (720 - self.pbp['Qtime']))

        # Format score so that it makes sense: 'XX-XX'
        self.pbp['SCORE'] = (self.pbp['SCORE']
                             .ffill()
                             .fillna('0 - 0'))
        return self

    def _get_player_ids(self):
        """
        Helper function for returning player ids for all players in game.
        Note: This data may also be somewhere more conveniently
            accessible in tracking_data.
        """
        ids = {}
        for index, row in self.pbp.iterrows():
            if row['PLAYER1_NAME'] not in ids:
                ids[row['PLAYER1_NAME']] = row['PLAYER1_ID']
            if row['PLAYER2_NAME'] not in ids:
                ids[row['PLAYER2_NAME']] = row['PLAYER2_ID']
            if row['PLAYER3_NAME'] not in ids:
                ids[row['PLAYER3_NAME']] = row['PLAYER3_ID']
        ids.pop(None, None)  # Remove None key if it exists, otherwise do nothing
        self.player_ids = ids
        return self

    def _get_player_jerseys(self):
        """
        Helper function for returning player jerseys for all players in game.
        """
        jerseys = {}
        # Combine home and away players
        # The structure is assumed to be self.tracking_data['events'][0]['home']['players'] based on standard SportVU data
        players = (self.tracking_data['events'][0]['home']['players'] +
                   self.tracking_data['events'][0]['visitor']['players'])
        for player in players:
            jerseys[player['playerid']] = player['jersey']
        self.player_jerseys = jerseys
        return self

    def _format_tracking_data(self):
        """
        Helper function to format tracking data into pandas DataFrame
        """
        events = pd.DataFrame(self.tracking_data['events'])
        moments = []
        # Extract 'moments': Each moment is an individual frame
        for row in events['moments']:
            for inner_row in row:
                moments.append(inner_row)
        moments = pd.DataFrame(moments)
        moments = moments.drop_duplicates(subset=[1])
        moments = moments.reset_index()

        moments.columns = ['index', 'quarter', 'universe_time', 'quarter_time',
                           'shot_clock', 'unknown', 'positions']
        moments['game_time'] = (moments.quarter - 1) * 720 + \
                               (720 - moments.quarter_time)
        moments.drop(['index', 'unknown'], axis=1, inplace=True)
        self.moments = moments
        return self

    def _draw_court(self, color="gray", lw=2, grid=False, zorder=0):
        """
        Helper function to draw court.
        Modified from Savvas Tjortjoglou with contribution
            from Michael Wheelock
        S. Tjortjoglou: http://savvastjortjoglou.com/nba-shot-sharts.html
        M. Wheelock: https://www.linkedin.com/in/michael-s-wheelock-a5635a66
        """
        line_color="black"
        floor_color="#EAD16E"
        general_floor_color="#C7A04C"
        paint_color="#D6606D"
        ax = plt.gca()

        # Create the court lines
        # outer = Rectangle((0, -50), width=94, height=50, facecolor=floor_color,
        #                   edgecolor=line_color, zorder=zorder, lw=lw)
        
        # l_hoop = Circle((5.35, -25), radius=.75, lw=lw, fill=False,
        #                 color=color, zorder=zorder)
        # r_hoop = Circle((88.65, -25), radius=.75, lw=lw, fill=False,
        #                 color=color, zorder=zorder)
        # l_backboard = Rectangle((4, -28), 0, 6, lw=lw, color=color,
        #                         zorder=zorder)
        # r_backboard = Rectangle((90, -28), 0, 6, lw=lw, color=color,
        #                         zorder=zorder)
        # l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, fill=False,
        #                         color=color, zorder=zorder)
        # l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, fill=False,
        #                         color=color, zorder=zorder)
        # r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, fill=False,
        #                         color=color, zorder=zorder)
        # r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, fill=False,
        #                         color=color, zorder=zorder)
        # l_free_throw = Circle((19, -25), radius=6, lw=lw, fill=False,
        #                       color=color, zorder=zorder)
        # r_free_throw = Circle((75, -25), radius=6, lw=lw, fill=False,
        #                       color=color, zorder=zorder)
        # l_corner_a = Rectangle((0, -3), 14, 0, lw=lw, color=color,
        #                        zorder=zorder)
        # l_corner_b = Rectangle((0, -47), 14, 0, lw=lw, color=color,
        #                        zorder=zorder)
        # r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=color,
        #                        zorder=zorder)
        # r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=color,
        #                        zorder=zorder)
        # l_arc = Arc((5, -25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
        #             color=color, zorder=zorder)
        # r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248,
        #             lw=lw, color=color, zorder=zorder)
        # half_court = Rectangle((47, -50), 0, 50, lw=lw, color=color,
        #                        zorder=zorder)
        # hc_big_circle = Circle((47, -25), radius=6, lw=lw, fill=False,
        #                        color=color, zorder=zorder)
        # hc_sm_circle = Circle((47, -25), radius=2, lw=lw, fill=False,
        #                       color=color, zorder=zorder)
        # court_elements = [l_hoop, l_backboard, l_outer_box, outer,
        #                   l_inner_box, l_free_throw, l_corner_a,
        #                   l_corner_b, l_arc, r_hoop, r_backboard,
        #                   r_outer_box, r_inner_box, r_free_throw,
        #                   r_corner_a, r_corner_b, r_arc, half_court,
        #                   hc_big_circle, hc_sm_circle]

        outer = Rectangle((0, -50), width=94, height=50, facecolor=general_floor_color,
                          edgecolor=line_color, zorder=zorder, lw=lw)

        # The "paint" areas with fill and outline
        l_outer_box = Rectangle((0, -33), 19, 16, lw=lw, facecolor=paint_color,
                                edgecolor=line_color, zorder=zorder+1)
        r_outer_box = Rectangle((75, -33), 19, 16, lw=lw, facecolor=paint_color,
                                edgecolor=line_color, zorder=zorder+1)

        # Inner center circle with fill
        hc_sm_circle = Circle((47, -25), radius=2, lw=lw, facecolor=paint_color,
                              edgecolor=line_color, zorder=zorder+1)

        # All other lines are just outlines (fill=False) with the line_color
        l_hoop = Circle((5.35, -25), radius=.75, lw=lw, fill=False,
                        color=line_color, zorder=zorder+1)
        r_hoop = Circle((88.65, -25), radius=.75, lw=lw, fill=False,
                        color=line_color, zorder=zorder+1)
        l_backboard = Rectangle((4, -28), 0, 6, lw=lw, color=line_color,
                                zorder=zorder+1)
        r_backboard = Rectangle((90, -28), 0, 6, lw=lw, color=line_color,
                                zorder=zorder+1)
        l_inner_box = Rectangle((0, -31), 19, 12, lw=lw, fill=False,
                                color=line_color, zorder=zorder+1)
        r_inner_box = Rectangle((75, -31), 19, 12, lw=lw, fill=False,
                                color=line_color, zorder=zorder+1)
        l_free_throw = Circle((19, -25), radius=6, lw=lw, fill=False,
                              color=line_color, zorder=zorder+1)
        r_free_throw = Circle((75, -25), radius=6, lw=lw, fill=False,
                              color=line_color, zorder=zorder+1)
        l_corner_a = Rectangle((0, -3), 14, 0, lw=lw, color=line_color,
                               zorder=zorder+1)
        l_corner_b = Rectangle((0, -47), 14, 0, lw=lw, color=line_color,
                               zorder=zorder+1)
        r_corner_a = Rectangle((80, -3), 14, 0, lw=lw, color=line_color,
                               zorder=zorder+1)
        r_corner_b = Rectangle((80, -47), 14, 0, lw=lw, color=line_color,
                               zorder=zorder+1)
        l_arc = Arc((5, -25), 47.5, 47.5, theta1=292, theta2=68, lw=lw,
                    color=line_color, zorder=zorder+1)
        r_arc = Arc((89, -25), 47.5, 47.5, theta1=112, theta2=248,
                    lw=lw, color=line_color, zorder=zorder+1)
        half_court = Rectangle((47, -50), 0, 50, lw=lw, color=line_color,
                               zorder=zorder+1)
        hc_big_circle = Circle((47, -25), radius=6, lw=lw, fill=False,
                               color=line_color, zorder=zorder+1)

        court_elements = [outer, l_outer_box, r_outer_box, hc_sm_circle,
                          l_hoop, l_backboard, l_inner_box, l_free_throw,
                          l_corner_a, l_corner_b, l_arc, r_hoop, r_backboard,
                          r_inner_box, r_free_throw, r_corner_a, r_corner_b,
                          r_arc, half_court, hc_big_circle]

        # Add the court elements onto the axes
        for element in court_elements:
            ax.add_patch(element)

        return ax

    def watch_play(self, game_time, length, highlight_player=None,
                   commentary=True, show_spacing=None, show_spacing_team=None,
                   show_velocity=False, show_control=None, use_time_control=False):
        """
        DEPRECIATED.  See animate_play() for similar (fastere) method

        Method for viewing plays in game.
        Outputs video file of play in {cwd}/temp

        Args:
            game_time (int): time in game to start video
                (seconds into the game).
                Currently game_time can also be an tuple of length
                two with (starting_frame, ending_frame) if you want
                to watch a play using frames instead of game time.
            length (int): length of play to watch (seconds)
            highlight_player (str): If not None, video will highlight
                the circle of the inputed player for easy tracking.
            commentary (bool): Whether to include play-by-play
                commentary underneath video
            show_spacing (str in ['home', 'away']): show convex hull
                of home or away team.
                if None, does not display any convex hull

        Returns: an instance of self, and outputs video file of play
        """
        warnings.warn(("watch_play is extremely slow. "
                       "Use animate_play for similar functionality, "
                       "but greater efficiency"))

        if type(game_time) == tuple:
            starting_frame = game_time[0]
            ending_frame = game_time[1]
        else:
            # Get starting and ending frame from requested game_time and length
            starting_frame = self.moments[self.moments.game_time.round() ==
                                          game_time].index.values[0]
            ending_frame = self.moments[self.moments.game_time.round() ==
                                        game_time + length].index.values[0]

        # Make video of each frame
        for frame in range(starting_frame, ending_frame):
            self.plot_frame(frame, highlight_player=highlight_player,
                            commentary=commentary, show_spacing=show_spacing,
                            show_spacing_team=show_spacing_team,
                            show_velocity=show_velocity, show_control=show_control,
                            use_time_control=use_time_control)
        
        ffmpeg_cmd = self._get_ffmpeg_path()
        command = [
            ffmpeg_cmd,
            '-framerate', '20',
            '-start_number', str(starting_frame),
            '-i', '%d.png',
            '-c:v', 'libx264',
            '-r', '30',
            '-pix_fmt', 'yuv420p',
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            f'{starting_frame}.mp4'
        ]
        
        print(f"Running ffmpeg command: {' '.join(command)}")
        try:
            subprocess.run(command, cwd='temp', check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating video: {e}")
        except FileNotFoundError:
            print(f"Error: ffmpeg executable not found at {ffmpeg_cmd}")

        # Delete images
        for file in os.listdir('./temp'):
            if os.path.splitext(file)[1] == '.png':
                os.remove('./temp/{file}'.format(file=file))

        return self

    def _get_ffmpeg_path(self):
        """
        Helper to find ffmpeg executable, specifically in Conda environments
        where it might not be in the system PATH.
        """
        # First check if it's in PATH
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        
        # Check standard Windows Conda location
        # sys.prefix points to the env root (e.g. C:\Users\user\anaconda3\envs\myenv)
        conda_ffmpeg = os.path.join(sys.prefix, 'Library', 'bin', 'ffmpeg.exe')
        if os.path.exists(conda_ffmpeg):
            print(f"Found ffmpeg in conda environment: {conda_ffmpeg}")
            return conda_ffmpeg
            
        return "ffmpeg"

    def animate_play(self, game_time, length, highlight_player=None,
                     commentary=True, show_spacing=None, show_spacing_team=None,
                     show_velocity=False, show_control=None, use_time_control=False):
        """
        Method for animating plays in game.
        Outputs video file of play in {cwd}/temp.
        Individual frames are streamed directly to ffmpeg without writing them
        to the disk, which is a great speed improvement over watch_play

        Args:
            game_time (int): time in game to start video
                (seconds into the game).
                Currently game_time can also be an tuple of length two
                with (starting_frame, ending_frame)if you want to
                watch a play using frames instead of game time.
            length (int): length of play to watch (seconds)
            highlight_player (str): If not None, video will highlight
                the circle of the inputed player for easy tracking.
            commentary (bool): Whether to include play-by-play commentary in
                the animation
            show_spacing (str) in ['home', 'away']: show convex hull
                spacing of home or away team.
                If None, does not show spacing.

        Returns: an instance of self, and outputs video file of play
        """
        if type(game_time) == tuple:
            starting_frame = game_time[0]
            ending_frame = game_time[1]
        else:
            # Get starting and ending frame from requested game_time and length
            starting_frame = self.moments[self.moments.game_time.round() ==
                                          game_time].index.values[0]
            ending_frame = self.moments[self.moments.game_time.round() ==
                                        game_time + length].index.values[0]

        # Make video of each frame
        filename = "temp/{game_time}.mp4".format(game_time=game_time)
        if commentary:
            size = (960, 960)
        else:
            size = (960, 480)
        ffmpeg_cmd = self._get_ffmpeg_path()
        
        cmdstring = (ffmpeg_cmd,
                     '-y', '-r', '20',  # fps
                     '-s', '%dx%d' % size,  # size of image string
                     '-pix_fmt', 'argb',  # Stream argb data from matplotlib
                     '-f', 'rawvideo',  '-i', '-',
                     '-vcodec', 'libx264', filename)

        print(f"Saving video to {os.path.abspath(filename)}...")
        # Stream plots to pipe
        pipe = Popen(cmdstring, stdin=PIPE)
        for frame in range(starting_frame, ending_frame):
            self.plot_frame(frame, highlight_player=highlight_player,
                            commentary=commentary, show_spacing=show_spacing,
                            show_spacing_team=show_spacing_team,
                            show_velocity=show_velocity, show_control=show_control,
                            use_time_control=use_time_control, pipe=pipe)
        pipe.stdin.close()
        pipe.wait()
        return self

    def watch_player_actions(self, player_name, action, length=15, max_vids=5):
        """
        Method for viewing all plays a player in the game had of a
        specified type.
        For example: all of Damian Lillards FG attempts in the game
        Outputs video file for each play in {cwd}/temp

        Args:
            player_name (str): Name of player for which to produce videos.
                Currently, player_name must be perfectly formatted and
                capitalized, since no string processing is performed.
            action (str) {'all_FG', 'made_FG', 'miss_FG', 'rebound'}:
                Action type of interest
            length (int): length of play to watch (seconds) for each action.
            max_vids (int): Maximum number of videos to produce.
                max_vids=None if all videos are desired.  If max_vids
                is less than the total number of actions in the game, the
                earliest actions are made into videos.

        Returns: an instance of self, and outputs video file of plays
        """
        player_action_times = self._get_player_actions(player_name, action)
        for index, time in enumerate(player_action_times):
            if index == max_vids:
                break
            # Use animate_play instead of watch_play for better efficiency
            try:
                print(f"Generating video {index+1}/{len(player_action_times) if max_vids is None else min(max_vids, len(player_action_times))}...")
                self.animate_play(time-length, length,
                                highlight_player=player_name,
                                commentary=True)
            except Exception as e:
                print(f"Error generating video for action at time {time}: {e}")
        return self

    def _get_commentary(self, game_time, commentary_length=6,
                        commentary_depth=10):
        """
        Helper function for returning play by play events for a
        given game time.

        Args:
            game_time (int): game time (in seconds) for which to
                retrieve commentary for
            commentary_length (int): Number of play-by-play calls to
                include in commentary
            commentary_depth (int): Number of seconds to look in past
                to retrieve play-by-play calls
                commentary_depth=10 looks at previous 10 seconds of
                game for play-by-play calls

        Returns: tuple of information (commentary_script, score)
            commentary_script (str): string of commentary
                Most recent play-by-play calls, seperated by line breaks
            score (str): Score at current time 'XX - XX'
        """
        commentary = []
        score = "0 - 0"
        for game_second in range(game_time - commentary_depth, game_time + 2):
            for index, row in self.pbp[self.pbp.game_time ==
                                       game_second].iterrows():
                # Check if descriptions are strings (not NaN)
                if isinstance(row['HOMEDESCRIPTION'], str):
                    commentary.append('{self.home_team}: '.format(self=self) +
                                      str(row['HOMEDESCRIPTION']))
                if isinstance(row['VISITORDESCRIPTION'], str):
                    commentary.append('{self.away_team}: '.format(self=self) +
                                      str(row['VISITORDESCRIPTION']))
                if isinstance(row['NEUTRALDESCRIPTION'], str):
                    commentary.append(str(row['NEUTRALDESCRIPTION']))
                score = str(row['SCORE'])
        
        # Pad with empty strings if fewer than commentary_length items
        while len(commentary) < commentary_length:
            commentary.append("")
            
        # Take only the last commentary_length items if we have too many
        if len(commentary) > commentary_length:
            commentary = commentary[-commentary_length:]
            
        commentary_script = "\n".join(commentary)
        return (commentary_script, score)

    def _get_player_actions(self, player_name, action):
        """
        Helper function to get all times a player performed a specific action

        Args:
            player_name (str): name of player to get all actions for
            action {'all_FG', 'made_FG', 'miss_FG', 'rebound'}:
                Type of action to get all times for.

        Returns:
            times (list): list of game times a player performed a
                specific specific action
        """
        player_id = self.player_ids[player_name]
        action_dict = {'all_FG': [1, 2], 'made_FG': [1],
                       'miss_FG': [2], 'rebound': [4]}
        action_df = self.pbp[(self.pbp['PLAYER1_ID'] == player_id) &
                             (self.pbp['EVENTMSGTYPE']
                              .isin(action_dict[action]))]
        times = list(action_df['game_time'])
        return times

    def _get_moment_details(self, frame_number, highlight_player=None):
        """
        Helper function for getting important information for a given frame

        Args:
            frame_number (int): Frame in game to retrieve data for
                frame_number gets player tracking data from
                    moments.iloc[frame_number]
            highlight_player (str): Name of player to be highlighted
                in downstream plotting.
                if None, no player is highlighted.

        Returns: tuple of data
            game_time (int): seconds into game of current moment
            x_pos (list): list of x coordinants for all players and ball
            y_pos (list): list of y coordinants for all players and ball
            colors (list): color coding of each player/ball for coordinant data
            sizes (list): size of each player/ball
                (used for showing ball height)
            quarter (int): Game quarter
            shot_clock (str): shot clock
            game_clock (str): game clock
            edges (list): list of marker edge sizes of each player for video.
                useful when trying to highlight a player by making
                their edge thicker.
            universe_time (int): Time in the universe, in msec
        """
        current_moment = self.moments.iloc[frame_number]
        game_time = int(np.round(current_moment['game_time']))
        universe_time = int(current_moment['universe_time'])
        x_pos, y_pos, colors, sizes, edges, jerseys = [], [], [], [], [], []
        # Get player positions
        for player in current_moment.positions:
            x_pos.append(player[2])
            y_pos.append(player[3])
            colors.append(self.team_colors[player[0]])
            # Use ball height for size (useful to sevie a shot)
            if player[0] == -1:
                sizes.append(max(150 - 2*(player[4] - 5)**2, 10))
            else:
                sizes.append(200)
            # highlight_player makes their outline much thicker on the video
            if (highlight_player and
                    player[1] == self.player_ids[highlight_player]):
                edges.append(5)
            else:
                edges.append(0.5)
            # Add jersey number
            if player[1] != -1:
                jerseys.append(self.player_jerseys[player[1]])
            else:
                jerseys.append(None)
        # Unfortunately, the plot is below the y axis,
        # so the y positions need to be corrected
        y_pos = np.array(y_pos) - 50
        shot_clock = current_moment.shot_clock
        if np.isnan(shot_clock):
            shot_clock = 24.00
        shot_clock = str(shot_clock).split('.')[0]
        game_min, game_sec = divmod(current_moment.quarter_time, 60)
        game_clock = "%02d:%02d" % (game_min, game_sec)
        quarter = current_moment.quarter
        return (game_time, x_pos, y_pos, colors, sizes, quarter,
                shot_clock, game_clock, edges, universe_time, jerseys)

    def plot_frame(self, frame_number, highlight_player=None,
                   commentary=True, show_spacing=False, show_spacing_team=None,
                   show_velocity=False, show_control=None, use_time_control=False,
                   plot_spacing=None, pipe=None):
        """
        Creates an individual the frame of game.
        Outputs .png file in {cwd}/temp

        Args:
            frame_number (int): number of frame in game to create
                frame_number gets player tracking data from
                moments.iloc[frame_number]
            highlight_player (str): Name of player to highlight
                (by making their outline thicker).
                if None, no player is highlighted
            commentary (bool): if True, add play-by-play commentary
                under frame
            show_spacing (str in ['vor', 'ch']): show convex hull or voronoi diagram
                of home or away team
                if None, does not display any
            show_spacing_team (str in  ['home','away']): show spacing for home or away team
                if None, does not display any
            show_velocity (bool): if True, show player velocity vectors
            show_control (str in ['home', 'away'] or bool): if specified, show space control heatmap
                for the given team. If True, defaults to offensive team.
            use_time_control (bool): if True and show_control is enabled, use physics-based
                time-to-reach instead of distance for space control calculation
            pipe (subprocesses.Popen): Popen object with open pipe
                to send image to if False, image is written to disk
                instead of sent to pipe

        Returns: an instance of self, and outputs .png file of frame
            If pipe, ARGB values are sent to pipe object instead of
            writing to disk.

        TODO be able to call this method by game time instead of frame_number
        """
        (game_time, x_pos, y_pos, colors, sizes,
         quarter, shot_clock, game_clock, edges,
         universe_time, jerseys) = self._get_moment_details(frame_number,
                                                   highlight_player=highlight_player)
        (commentary_script, score) = self._get_commentary(game_time)
        fig = plt.figure(figsize=(12, 6), dpi=80)
        self._draw_court()
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticks([])
        frame.axes.get_yaxis().set_ticks([])
        plt.scatter(x_pos, y_pos, c=colors, s=sizes, alpha=0.85,
                    linewidths=edges)
        
        # Add jersey numbers
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            if jerseys[i]:
                plt.text(x, y, jerseys[i], ha='center', va='center',
                         color='white', fontsize=10, fontweight='bold')

        plt.xlim(-5, 100)
        plt.ylim(-55, 5)
        sns.set_style('dark')
        if commentary:
            plt.figtext(0.23, -.6, commentary_script, size=20)
        plt.figtext(0.43, 0.125, shot_clock, size=18)
        plt.figtext(0.5, 0.125, 'Q'+str(quarter), size=18)
        plt.figtext(0.57, 0.125, str(game_clock), size=18)
        plt.figtext(0.43, .85,
                    self.away_team + "  " + score + "  " + self.home_team,
                    size=18)
        if highlight_player:
            plt.figtext(0.17, 0.85, highlight_player, size=18)
        # Add team color indicators to top of frame
        plt.scatter([30, 67], [2.5, 2.5], s=100,
                    c=[self.team_colors[self.away_id],
                       self.team_colors[self.home_id]])
        if show_spacing == "ch":
            # Show convex hull on frame
            xy_pos = np.column_stack((np.array(x_pos), np.array(y_pos)))
            if show_spacing_team == 'home':
                points = xy_pos[1:6, :]
            if show_spacing_team == 'away':
                points = xy_pos[6:, :]
            hull = ConvexHull(points)
            hull_points = points[hull.vertices, :]
            polygon = Polygon(hull_points, alpha=0.3, color='gray')
            ax = plt.gca()
            ax.add_patch(polygon)
        
        if show_spacing == 'vor':
            # Show Voronoi diagram on frame
            details = self._get_moment_details(frame_number)
            x_pos = np.array(details[1])
            y_pos = np.array(details[2])
            
            if len(x_pos) == 11:
                player_x = x_pos[1:]
                player_y = y_pos[1:]
                players = np.column_stack((player_x, player_y))
                
                # Use mirroring to bound the cells
                mirrors = []
                for p in players:
                    mirrors.append([-p[0], p[1]])
                    mirrors.append([2*94 - p[0], p[1]])
                    mirrors.append([p[0], -p[1]])
                    mirrors.append([p[0], -100 - p[1]])
                
                points = np.concatenate([players, mirrors])
                vor = Voronoi(points)
                ax = plt.gca()
                
                for i in range(10):
                    region_idx = vor.point_region[i]
                    region_vertices_indices = vor.regions[region_idx]
                    
                    if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
                        region_vertices = vor.vertices[region_vertices_indices]
                        color = 'red' if i < 5 else 'blue'
                        polygon = Polygon(region_vertices, alpha=0.2, facecolor=color, edgecolor='black', lw=1)
                        ax.add_patch(polygon)
        
        if show_velocity and frame_number > 0:
            # Calculate and plot player velocities using refined physics
            physics = self.get_player_physics(frame_number)
            
            if physics:
                curr = self._get_moment_details(frame_number)
                x = np.array(curr[1][1:])
                y = np.array(curr[2][1:])
                
                # Extract vectors and speeds
                dx = np.array([physics[i]['velocity'][0] for i in range(1, 11)])
                dy = np.array([physics[i]['velocity'][1] for i in range(1, 11)])
                speeds = np.array([physics[i]['speed'] for i in range(1, 11)])

                circle_offset = 1.8  # Pushes arrow start to the edge of the circle
                min_visual_speed = 2.0 # Ensures slow players have a visible arrow
                
                # Create copies for plotting so we don't overwrite raw data
                plot_dx = dx.copy()
                plot_dy = dy.copy()
                
                moving_mask = speeds > 0.1
                
                # Normalize arrows for slow players so they aren't 'stumps'
                # but keep the direction intact
                for i in range(len(speeds)):
                    if 0.1 < speeds[i] < min_visual_speed:
                        plot_dx[i] = (dx[i] / speeds[i]) * min_visual_speed
                        plot_dy[i] = (dy[i] / speeds[i]) * min_visual_speed

                x_start = x.copy()
                y_start = y.copy()
                
                # Shift the start point to the perimeter of the s=200 circle
                x_start[moving_mask] += (dx[moving_mask] / speeds[moving_mask]) * circle_offset
                y_start[moving_mask] += (dy[moving_mask] / speeds[moving_mask]) * circle_offset

                # Render arrows
                plt.quiver(x_start, y_start, plot_dx, plot_dy, 
                           angles='xy', scale_units='xy', scale=5, 
                           color='black', width=0.005, headwidth=3, 
                           headlength=5, zorder=5, pivot='tail')
                
                # Render text labels slightly further out
                for i, (px, py, speed) in enumerate(zip(x, y, speeds)):
                    # Use the original dx/dy to decide where to put the text
                    # so it doesn't overlap the arrow itself
                    text_x = px + (dx[i]/speeds[i] * 3) if speeds[i] > 0.1 else px + 1
                    text_y = py + (dy[i]/speeds[i] * 3) if speeds[i] > 0.1 else py + 1
                    
                    plt.text(text_x, text_y, f"{speed:.1f} ft/s", 
                             fontsize=8, color='black', alpha=0.8, 
                             fontweight='bold', ha='center')
                
        if show_control:
            # Determine which team's perspective to show
            control_team = show_control if isinstance(show_control, str) else 'home'
            
            X, Y, Z = self.get_space_control(frame_number, team=control_team, 
                                             resolution=50, use_time=use_time_control)
            
            if X is not None:
                # Plot heatmap with diverging colormap
                # Positive values (red) = control_team controls space
                # Negative values (blue) = opponent controls space
                ax = plt.gca()
                im = ax.imshow(Z, extent=[0, 94, -50, 0], origin='lower',
                              cmap='RdBu_r', alpha=0.5, vmin=-12, vmax=12, zorder=1)
                
        if pipe:
            # Write ARGB values to pipe
            fig.canvas.draw()
            string = fig.canvas.tostring_argb()
            pipe.stdin.write(string)
            plt.close()
            if commentary:
                fig = plt.figure(figsize=(12, 6), dpi=80)
                plt.figtext(.2, .4, commentary_script, size=20)
                fig.canvas.draw()
                string = fig.canvas.tostring_argb()
                pipe.stdin.write(string)
            plt.close()

        else:
            # Save image to disk
            plt.savefig('temp/{frame_number}.png'
                        .format(frame_number=frame_number),
                        bbox_inches='tight')
            plt.close()
        return self

    def _in_formation(self, frame_number):
        """
        This is a complicated method to explain, but it is actually
        very simple.
        It determines if the game is in a set offense/defense.
        It basically returns True if a normal play is being run,
        and False if the game is in transition, out of bounds,
        free throw, etc.  It is useful for analyzing plays that teams
        run, and discarding all extranous times from the game.
        """
        # Get relevant moment details
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        shot_clock = details[6]
        # Determine if offense/defense is set
        if float(shot_clock) < 23:
            if (x_pos < 47).all() or (x_pos > 47).all():
                return True
        return False

    def get_spacing_area(self, frame_number):
        """
        Calculates convex hull of home and away team for a given frame.
        Useful for analyzing the spacing of teams.

        Args:
            frame_number (int): number of frame in game to calculate
                team convex hulls

        Returns: tuple of data (home_area, away_area)
            home_area (float): convex hull area of home team
            away_area (float): convex hull area of away team

        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        y_pos = np.array(details[2])
        xy_pos = np.column_stack((x_pos, y_pos))
        home_area = ConvexHull(xy_pos[1:6, :]).area
        away_area = ConvexHull(xy_pos[6:, :]).area
        return (home_area, away_area)

    def get_voronoi_areas(self, frame_number):
        """
        Calculates Voronoi cells for each player and returns the total area
        occupied by the home and away teams, clipped to the court boundaries.

        Args:
            frame_number (int): number of frame in game to calculate
                team Voronoi areas

        Returns: tuple of data (home_voronoi_area, away_voronoi_area)
            home_voronoi_area (float): total area occupied by home team
            away_voronoi_area (float): total area occupied by away team
        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        y_pos = np.array(details[2])
        
        # We only care about the 10 players, not the ball (index 0 is ball)
        if len(x_pos) < 11:
            return (0.0, 0.0)
            
        player_x = x_pos[1:]
        player_y = y_pos[1:]
        players = np.column_stack((player_x, player_y))
        
        # Define court boundaries
        # x: [0, 94], y: [-50, 0]
        
        # Mirroring technique to clip Voronoi cells to the rectangle
        mirrors = []
        for p in players:
            mirrors.append([-p[0], p[1]]) # Mirror across x=0
            mirrors.append([2*94 - p[0], p[1]]) # Mirror across x=94
            mirrors.append([p[0], -p[1]]) # Mirror across y=0
            mirrors.append([p[0], -100 - p[1]]) # Mirror across y=-50
            
        points = np.concatenate([players, mirrors])
        vor = Voronoi(points)
        
        home_area = 0.0
        away_area = 0.0
        
        for i in range(10):
            region_idx = vor.point_region[i]
            region_vertices_indices = vor.regions[region_idx]
            
            if -1 not in region_vertices_indices and len(region_vertices_indices) > 0:
                region_vertices = vor.vertices[region_vertices_indices]
                cell_area = ConvexHull(region_vertices).volume 
                
                if i < 5: # Home team
                    home_area += cell_area
                else: # Away team
                    away_area += cell_area
                    
        return (home_area, away_area)

    def _time_to_reach(self, player_pos, player_vel, target_pos, max_accel=10.0, max_speed=22.0):
        """
        Calculates minimum time for a player to reach a target position.
        Uses kinematic equations accounting for current velocity, acceleration, and max speed.
        
        Args:
            player_pos (np.array): Current position [x, y]
            player_vel (np.array): Current velocity [vx, vy] in ft/s
            target_pos (np.array): Target position [x, y]
            max_accel (float): Maximum acceleration in ft/s²
            max_speed (float): Maximum speed in ft/s
            
        Returns:
            float: Minimum time to reach target in seconds
        """
        # Vector from player to target
        displacement = target_pos - player_pos
        distance = np.linalg.norm(displacement)
        
        if distance < 0.1:  # Already at target
            return 0.0
            
        # Unit vector toward target
        direction = displacement / distance
        
        # Current speed and velocity component toward target
        current_speed = np.linalg.norm(player_vel)
        v_parallel = np.dot(player_vel, direction)  # Velocity toward target
        
        # Simplified model: assume player can instantly redirect velocity toward target
        # More realistic would include turning radius, but this is computationally simpler
        
        # If already at or above max speed
        if current_speed >= max_speed:
            return distance / max_speed
        
        # Distance to accelerate to max speed from current speed in target direction
        # v² = v₀² + 2ad  =>  d = (v² - v₀²) / (2a)
        v_initial = max(0, v_parallel)  # Only count positive component
        
        if v_initial >= max_speed:
            # Already at max speed toward target
            return distance / max_speed
        
        # Distance needed to reach max speed
        accel_distance = (max_speed**2 - v_initial**2) / (2 * max_accel)
        
        if accel_distance >= distance:
            # Won't reach max speed, just accelerate the whole way
            # d = v₀t + ½at²  =>  solve quadratic
            # ½at² + v₀t - d = 0
            a, b, c = 0.5 * max_accel, v_initial, -distance
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                # Fallback to simple distance/speed
                return distance / max(current_speed, 1.0)
            t = (-b + np.sqrt(discriminant)) / (2*a)
            return t
        else:
            # Accelerate to max speed, then cruise
            # Time to accelerate: v = v₀ + at  =>  t = (v - v₀)/a
            t_accel = (max_speed - v_initial) / max_accel
            
            # Remaining distance at max speed
            remaining_distance = distance - accel_distance
            t_cruise = remaining_distance / max_speed
            
            return t_accel + t_cruise

    def get_space_control(self, frame_number, team='home', resolution=50, use_time=False):
        """
        Calculates space control heatmap using delta-distance or delta-time metric.
        For each point on the court, computes:
        - If use_time=False: delta_d(x,y) = d_closest_opponent - d_closest_teammate
        - If use_time=True: delta_t(x,y) = t_closest_opponent - t_closest_teammate
        
        Args:
            frame_number (int): Frame number to analyze
            team (str): 'home' or 'away' - perspective for control calculation
            resolution (int): Grid resolution (higher = more detail, slower)
            use_time (bool): If True, use physics-based time-to-reach instead of distance
            
        Returns:
            tuple: (X, Y, Z) where X and Y are meshgrid coordinates and
                   Z is the control values (positive = team controls, negative = opponent controls)
        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        y_pos = np.array(details[2])
        
        if len(x_pos) < 11:
            return None, None, None
            
        # Separate teams (skip ball at index 0)
        home_x = x_pos[1:6]
        home_y = y_pos[1:6]
        away_x = x_pos[6:11]
        away_y = y_pos[6:11]
        
        # Get velocities if using time-based metric
        if use_time:
            physics = self.get_player_physics(frame_number)
            if not physics:
                # Fallback to distance-based if physics unavailable
                use_time = False
            else:
                home_vels = [physics[i]['velocity'] for i in range(1, 6)]
                away_vels = [physics[i]['velocity'] for i in range(6, 11)]
        
        # Create grid across court
        x_grid = np.linspace(0, 94, resolution)
        y_grid = np.linspace(-50, 0, resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize control array
        Z = np.zeros_like(X)
        
        # Determine which team we're calculating for
        if team == 'home':
            team_x, team_y = home_x, home_y
            opp_x, opp_y = away_x, away_y
            if use_time:
                team_vels = home_vels
                opp_vels = away_vels
        else:
            team_x, team_y = away_x, away_y
            opp_x, opp_y = home_x, home_y
            if use_time:
                team_vels = away_vels
                opp_vels = home_vels
        
        # Calculate delta_d or delta_t for each grid point
        for i in range(resolution):
            for j in range(resolution):
                target = np.array([X[i, j], Y[i, j]])
                
                if use_time:
                    # Time-based: calculate time to reach for each player
                    team_times = []
                    for k in range(5):
                        pos = np.array([team_x[k], team_y[k]])
                        vel = team_vels[k]
                        t = self._time_to_reach(pos, vel, target)
                        team_times.append(t)
                    
                    opp_times = []
                    for k in range(5):
                        pos = np.array([opp_x[k], opp_y[k]])
                        vel = opp_vels[k]
                        t = self._time_to_reach(pos, vel, target)
                        opp_times.append(t)
                    
                    t_teammate = np.min(team_times)
                    t_opponent = np.min(opp_times)
                    
                    # Delta time (positive = team controls, negative = opponent controls)
                    Z[i, j] = t_opponent - t_teammate
                else:
                    # Distance-based (original implementation)
                    team_dists = np.sqrt((team_x - target[0])**2 + (team_y - target[1])**2)
                    d_teammate = np.min(team_dists)
                    
                    opp_dists = np.sqrt((opp_x - target[0])**2 + (opp_y - target[1])**2)
                    d_opponent = np.min(opp_dists)
                    
                    # Delta distance (positive = team controls, negative = opponent controls)
                    Z[i, j] = d_opponent - d_teammate
                
        return X, Y, Z

    def get_distance(self, frame_number, highlight_player):
        """
        Placeholder for distance calculation logic.
        """
        pass
    def get_offensive_team(self, frame_number):
        """
        Determines which team is on offense.
        Currently only works if team is in set offense or defense.

        Args:
            frame_number (int): number of frame in game to determine
                offensive team

        Returns:
            str in ['home', 'away']
        """
        details = self._get_moment_details(frame_number)
        x_pos = np.array(details[1])
        quarter = details[5]
        if len(x_pos) != 11:
            return None
        if self.flip_direction:
            if (x_pos < 47).all() and quarter in [1, 2]:
                return 'away'
            if (x_pos > 47).all() and quarter in [3, 4]:
                return 'away'
            if (x_pos < 47).all() and quarter in [3, 4]:
                return 'home'
            if (x_pos > 47).all() and quarter in [1, 2]:
                return 'home'
        if (x_pos < 47).all() and quarter in [1, 2]:
            return 'home'
        if (x_pos > 47).all() and quarter in [3, 4]:
            return 'home'
        if (x_pos < 47).all() and quarter in [3, 4]:
            return 'away'
        if (x_pos > 47).all() and quarter in [1, 2]:
            return 'away'
        return None

    def _determine_direction(self):
        """
        Helper funcation to determine which direction the home team is going.
        Surprisingly, this is not consistent and depends on the game.
        Currently, this method detects which side the players start on and is
        ~90% accurate
        """
        incorrect_count = 0
        correct_count = 0
        for frame in range(0, 10000, 100):
            details = self._get_moment_details(frame)
            home_team_x = details[1][1:6]
            away_team_x = details[1][6:]
            if np.mean(home_team_x) < np.mean(away_team_x):
                incorrect_count += 1
            else:
                correct_count += 1
        if incorrect_count > correct_count:
            self.flip_direction = True
        return None

    def get_player_physics(self, frame_number):
        """
        Calculates kinematic properties (velocity and acceleration) for all players.
        Units are in ft/s and ft/s^2.
        
        Args:
            frame_number (int): number of frame in game to calculate physics for
            
        Returns: dict with player indices as keys and (velocity, acceleration) as values.
                 Velocity and acceleration are numpy arrays [vx, vy] and [ax, ay].
        """
        if frame_number < 1 or frame_number >= len(self.moments) - 1:
            return {}

        # We need three frames to get acceleration comfortably (central difference or forward/backward)
        # Let's use current and previous for velocity, and previous/current/next for acceleration
        curr = self._get_moment_details(frame_number)
        prev = self._get_moment_details(frame_number - 1)
        next_m = self._get_moment_details(frame_number + 1)
        
        # Universe time is in milliseconds
        dt1 = (curr[9] - prev[9]) / 1000.0 # Time between prev and curr in seconds
        dt2 = (next_m[9] - curr[9]) / 1000.0 # Time between curr and next in seconds
        
        if dt1 == 0 or dt2 == 0:
            return {}
            
        physics = {}
        for i in range(1, 11): # Skip ball
            # Positions
            p_prev = np.array([prev[1][i], prev[2][i]])
            p_curr = np.array([curr[1][i], curr[2][i]])
            p_next = np.array([next_m[1][i], next_m[2][i]])
            
            # Velocities (ft/s)
            v1 = (p_curr - p_prev) / dt1
            v2 = (p_next - p_curr) / dt2
            
            # Instantaneous velocity at current frame (average of v1 and v2 is common)
            v_curr = (v1 + v2) / 2.0
            
            # Acceleration (ft/s^2)
            # a = (v2 - v1) / ((dt1 + dt2) / 2)
            a_curr = (v2 - v1) / ((dt1 + dt2) / 2.0)
            
            physics[i] = {
                'velocity': v_curr,
                'acceleration': a_curr,
                'speed': np.linalg.norm(v_curr),
                'accel_mag': np.linalg.norm(a_curr)
            }
            
        return physics

    def get_frame(self, game_time):
        """
        Converts a game time to a frame number.  Useful all over the place.

        Args:
            game_time (int): game time in seconds of interest

        Returns:
            frame (int): frame number of game time
        """
        test_time = game_time
        while True:
            if test_time in self.moments.game_time.round():
                frames = self.moments[self.moments.game_time.round() ==
                                      test_time].index.values
                if len(frames) > 0:
                    frame = frames[0]
                    break
                else:
                    test_time -= 1
            else:
                test_time -= 1
        return frame

    def get_play_frames(self, event_num, play_type='offense'):
        """
        Args:
            event_num (int): EVENTNUM of interest in games.pbp
                NOTE: Check pbpevents.txt for event numbers
            play_type (str in ['offense', 'defense']): Team of interest
                is offense or defense

        Returns:
            tuple of (start_time (int), end_time (int)): start time
                and end time in seconds for play of interest
        """
        play_index = self.pbp[self.pbp['EVENTNUM'] == event_num].index[0]
        event_team = str(self.pbp[self.pbp['EVENTNUM'] == event_num]
                         .PLAYER1_TEAM_ABBREVIATION.head(1).values[0])
        if event_team == self.home_team:
            target_team = 'home'
        if event_team == self.away_team:
            target_team = 'away'
        end_time = int(self.pbp[self.pbp['EVENTNUM'] == event_num].game_time)
        # To find lower bound on starting frame of the play,
        # determining when previous play ended
        putative_start_time = int(self.pbp.iloc[play_index-1].game_time)
        putative_start_frame = self.get_frame(putative_start_time)
        end_frame = self.get_frame(end_time)
        for test_frame in range(putative_start_frame, end_frame):
            if self.get_offensive_team(test_frame) == target_team:
                break
        # If the previous loop never found an offensive play,
        # the function returns None
        else:
            return None
        # Add two seconds to game time to let the players settle into position
        start_frame = self.get_frame(round(self.moments.iloc[test_frame].game_time + 2))
        return (start_frame, end_frame)
