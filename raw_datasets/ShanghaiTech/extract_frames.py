import cv2
import os


def get_frames(videoFolder):
    videoList = os.listdir(videoFolder)  # Get all video names.
    videoList.sort()  # Sort videos by number.
    
    frameFolder = videoFolder + 'Frame'  # The folder name that holds all video frames. 
    
    for v in videoList:
        # Create a folder with the same name for each video.
        volFolder = str(v)
        os.makedirs(frameFolder+"/"+volFolder, exist_ok=True)
        
        # Read videos.
        videoPath = videoFolder + '/' + v
        vidcap = cv2.VideoCapture(videoPath)
        numFrame = 0
        success, frame = vidcap.read()
        
        # Store video frames.
        while success:
            frameName = ("%06d.jpg" % numFrame)
            framePath = frameFolder+"/"+volFolder+"/"+frameName
            cv2.imwrite(framePath, frame)
            success, frame = vidcap.read()
            numFrame = numFrame + 1
            print('extracting {}-th frame of video {}'.format(numFrame, v))
        
        vidcap.release()


if __name__ == '__main__':
    get_frames('./training/videos')

