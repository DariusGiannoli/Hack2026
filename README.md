1. On your PC — copy updated sender to Orin:


scp ~/Desktop/haptic_teleoperation/robot/g1_stream_sender.py unitree@192.168.123.164:~/
2. On the Orin` — start sending:


sudo python3.8 ~/g1_stream_sender.py 192.168.123.100 9000 30
3. On your PC (new terminal) — start receiving:


conda run -n haptic python3 ~/Desktop/haptic_teleoperation/robot/g1_stream_receiver.py 9000