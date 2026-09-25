"""Compose three Blender skill clips and a labeled poster on #FDFAF4."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
skills=[('walk','Repositioning','repositioning.mp4',91),('dribble','Dribbling','dribbling.mp4',91),('shoot','Kicking','kicking.mp4',22)]
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
filters=[]
for i,(_,label,_,_) in enumerate(skills):
    filters.append(f'[{i}:v]scale=480:320,pad=512:392:16:12:color=0xFDFAF4,drawbox=x=16:y=12:w=480:h=320:color=0xE4E1DA:t=1,drawtext=fontfile={font}:text={label}:fontcolor=0x6E747C:fontsize=20:x=(w-tw)/2:y=347[p{i}]')
filters.append('[p0][p1][p2]hstack=inputs=3[v]')
video=['ffmpeg','-y','-loglevel','error']
poster=['ffmpeg','-y','-loglevel','error']
for skill,_,filename,frame in skills:
    video+=['-i',str(out/skill/filename)]
    poster+=['-i',str(out/skill/f'preview_{frame:04}.png')]
video+=['-filter_complex',';'.join(filters),'-map','[v]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'skills_comparison.mp4')]
poster+=['-filter_complex',';'.join(filters),'-map','[v]','-frames:v','1',str(out/'skills_comparison.png')]
subprocess.run(video,check=True);subprocess.run(poster,check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames:format=duration','-of','json',str(out/'skills_comparison.mp4')]))
assert probe['streams'][0]['nb_read_frames']=='180' and float(probe['format']['duration'])==6
(out/'comparison_verification.json').write_text(json.dumps(probe,indent=2))
print('SKILL_PANELS_VERIFIED',probe)
