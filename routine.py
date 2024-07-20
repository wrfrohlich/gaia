from load import GaitLab, GWalk

gait_lab_file = "../bittencourt_data/force_track_040102.emt"
gwalk_file = "../bittencourt_data/gwalk_040202.txt"

#GL = GaitLab()
#gait_lab_data = GL.load_data(gait_lab_file)

GW = GWalk()
gwalk_data = GW.load_data(gwalk_file)
