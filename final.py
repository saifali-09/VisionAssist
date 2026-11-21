import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import speech_recognition as sr
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


wCam,hCam=640,480
mid=320
action,obj=None,None
midHeight=240
direction=''
notfound=True

model = YOLO("yolov8n.pt")  

en = pyttsx3.init()
en.setProperty('rate', 150)
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(direction,object):
    
    if direction == 'right':
        engine.say(f"{object} in right")
    elif direction == 'left':
        engine.say(f"{object} in left")
    elif direction=='centered':
        engine.say(f"{object} in front of you")
    elif direction=='notFound':
        engine.say(f"searching {object}")
    else:
        print("not found")    

    engine.runAndWait()







def countSpeak(obje,a):
    # count=int(obje[0])
    if obje[0] !=None:
        print(obje[0])
        count=int(obje[0])
        if count>1:
            en.say(f'there are{count} {obje[1]}')
        elif count ==1:
            en.say(f'there is {count} {obje[1]}')     
        else:
            en.say(f"{object} not found")
        
    elif obje[2] !=None:
        en.say(f"scanning for {obje[2]} seconds")
        en.say(f"scanning for {obje[2]} seconds")


    en.runAndWait()






def detect_objects(frame):
   
    results = model(frame)
    # if count==1:
    #     print(model(frame)[0])
    #     count+1
    return results

def draw_detections(frame, results,findingObject):
   
    spoken = False  
    global notfound
    # print(results)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = model.names[cls]
            
            if label ==findingObject  and conf > 0.5:
                notfound=False
                direction = ''
                if x1 > mid + 50:
                    direction = 'right'
                elif x2 < mid - 50:
                    direction = 'left'
                else:
                    direction = 'centered'

                if not spoken:
                    threading.Thread(target=speak, args=(direction,findingObject), daemon=True).start()
                    spoken = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(frame, (mid, midHeight), (mid + 5, midHeight + 5), (255, 255, 0), -1)

               
                cv2.putText(frame, label + ' ' + direction, (x1 + 10, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                if not spoken and notfound==True:
                    print(notfound)
                    threading.Thread(target=speak, args=("notFound",findingObject), daemon=True).start()
                    spoken = True


    return frame

def extract_command(text):
    text = text.lower()
    words = text.split()

    if "find" in words:
        idx = words.index("find")
        if idx + 1 < len(words):
            return "find", words[idx + 1]
    elif "how" in words and "many" in words:
        idx = words.index("many")
        if idx + 1 < len(words):
            return "count", words[idx + 1]

    return None, None  

def main():
  
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        r.adjust_for_ambient_noise(source, duration=2)
        r.energy_threshold = 100
        print("Listening for command...")

        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=5)
            print("Recognizing...")

            text = r.recognize_google(audio)
            print("You said:", text)

            action, obj = extract_command(text)

            if action and obj:
                print(f"Action: {action}")
                print(f"Object: {obj}")
            else:
                print("Invalid command format. Please say 'find <object>' or 'how many <object>'.")

        except sr.WaitTimeoutError:
            print("Timed out waiting for speech.")
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError as e:
            print(f"API error: {e}")



    ########################################################
    if action:
        cap = cv2.VideoCapture(0)  
        cap.set(3,wCam)
        cap.set(4,hCam)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        if action=='find' and obj:
            while True:
                ret, frame = cap.read()

                # frame=cv2.flip(frame,1)
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                results = detect_objects(frame)
                frame = draw_detections(frame, results,obj)
                
                cv2.imshow("YOLOv8 Real-Time Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



        elif action=='count' and obj:
            model = YOLO("yolov8n.pt")
            tracker = DeepSort(
            max_age=50,         
            n_init=10,          
            max_cosine_distance=0.4
            )
          
            unique_person_ids = set()
           
            start_time = time.time()
            duration = 30 # seconds
            # scanning(duration)
            scanningcount=0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if scanningcount<1:
                    threading.Thread(target=countSpeak,args=([None,None,'30'],''),daemon=True).start()
                    scanningcount=scanningcount+1

                
                results = model(frame)[0]
                detections = []

                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                   
                    if label == obj and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        detections.append(([x1, y1, w, h], conf, label))

                
                tracks = tracker.update_tracks(detections, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id

                   
                    if track_id not in unique_person_ids:
                        unique_person_ids.add(track_id)

                   
                    l, t, w, h = track.to_ltrb()
                    cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

               
                cv2.imshow("Unique Person Tracker", frame)

              
                if time.time() - start_time > duration:
                    print(f"✅ Done! Unique people detected: {len(unique_person_ids)}")
                    # threading.Thread(target=countSpeak, args=(len(unique_person_ids),obj), daemon=True).start()
                    # countSpeak(len(unique_person_ids),obj)
                    threading.Thread(target=countSpeak, args=([str(len(unique_person_ids)),obj,None],''),daemon=True).start()


                    break

                # Press 'q' to quit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(5)
            
    
    

if __name__ == "__main__":
    main()

# 100 paar
    
#     while True:
#         ret, frame = cap.read()
#         # frame=cv2.flip(frame,1)
#         if not ret:
#             print("Error: Failed to capture frame.")
#             break
        
#         results = detect_objects(frame)
#         frame = draw_detections(frame, results)
        
#         cv2.imshow("YOLOv8 Real-Time Detection", frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()




# 1.voice of say something
# 2. if not say find or how many that start again 
#     saying unknown thing
# 3. improve finding if there are more than one object