"""
EMG recording script for two sensors on ESP32 (pins 34 & 35) via serial (COM5 by default).

Features implemented per user request:
- Records each repetition for 3 seconds, repeated 15 times (total 45s per gesture).
- GESTURES mapping as requested.
- Prints clear terminal instructions during recording.
- Saves combined data for each gesture into CSV files under `data_raw/`.
- Live plotting (Matplotlib) with two subplots (Sensor 1 top, Sensor 2 bottom) that show the full 45s
  of data while recording. Uses deque buffers for smooth updates.
- Saves plot images into `graphs/`.

Notes:
- Requires pyserial, pandas, matplotlib, numpy. Install with: pip install pyserial pandas matplotlib numpy
- Set PORT and BAUD to match your ESP32 settings.

Usage: run the script and follow the terminal prompts.
"""

import os
import time
import re
from collections import deque
import serial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# CONFIGURATION
# =========================
PORT = "COM5"          # ⚠️ Change this to your ESP32 port if different
BAUD = 115200
DURATION_PER_REP = 3     # seconds per repetition (user requested 3s)
REPETITIONS = 15         # repetitions per gesture (user requested 15)
TOTAL_DURATION = DURATION_PER_REP * REPETITIONS  # 45 seconds total
SAVE_DIR = "data_raw"
GRAPH_DIR = "graphs"

GESTURES = {
	"rest": "Neutral / no message",
	"fist": "Yes / OK",
	"open": "No / Stop",
	"wrist_up": "Hello / Start speaking",
	"wrist_down": "Goodbye / End speaking",
	"strong_fist": "Urgent / Need help"
}

# How frequently to update the plot (seconds). Lower -> smoother but more CPU.
PLOT_UPDATE_INTERVAL = 0.05  # 50 ms


# =========================
# HELPERS
# =========================

def parse_dual_channel(line):
	"""
	Extract two integers from a serial line.
	Accepts formats like: '512,478' or 'EMG1:512 EMG2:478' or '512 478'.
	Returns tuple (int, int) or (None, None) if parsing fails.
	"""
	nums = re.findall(r"\d+", line)
	if len(nums) >= 2:
		try:
			return int(nums[0]), int(nums[1])
		except ValueError:
			return None, None
	return None, None


def ensure_dirs():
	os.makedirs(SAVE_DIR, exist_ok=True)
	os.makedirs(GRAPH_DIR, exist_ok=True)


def adc_to_volt(adc_value, vref=3.3, resolution=4095):
	"""Convert ADC integer (0..resolution) to voltage (V)."""
	try:
		return (adc_value / resolution) * vref
	except Exception:
		return 0.0


def record_gesture(ser, gesture_name):
	"""
	Record one gesture for REPETITIONS × DURATION_PER_REP seconds.

	This function performs:
	- Prompts for each repetition.
	- Collects data into per-gesture buffers (list of rows).
	- Shows a live plot that fills across the full 45s window.
	- Saves combined CSV for the gesture and saves the final plot image.
	"""

	print(f"\n=== Gesture: {gesture_name.upper()} ===")
	print(f"Meaning: {GESTURES.get(gesture_name, 'N/A')}")

	# Prepare buffers that will hold the entire 45s of data for this gesture
	timestamps = deque()  # seconds since start of first rep for this gesture
	emg1 = deque()
	emg2 = deque()
	rows = []  # list to store CSV rows (ms, adc_ch1, volt_ch1, adc_ch2, volt_ch2, rep)

	# Prepare plotting
	plt.ion()
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
	fig.suptitle(f"EMG Recording — {gesture_name} — {REPETITIONS} reps × {DURATION_PER_REP}s = {TOTAL_DURATION}s")

	ax1.set_ylabel("EMG Sensor 1 Signal (ADC)")
	ax1.grid(True)
	line1, = ax1.plot([], [], color="C0", label="EMG Sensor 1")
	ax1.legend(loc="upper right")

	ax2.set_xlabel("Time (s)")
	ax2.set_ylabel("EMG Sensor 2 Signal (ADC)")
	ax2.grid(True)
	line2, = ax2.plot([], [], color="C1", label="EMG Sensor 2")
	ax2.legend(loc="upper right")

	# Fix x-axis to the full total duration
	ax2.set_xlim(0, TOTAL_DURATION)

	overall_start = None

	try:
		# Single prompt to start continuous recording for all repetitions
		input(f"\n👉 Prepare for '{gesture_name}'. Press Enter to start continuous recording ({REPETITIONS} reps × {DURATION_PER_REP}s = {TOTAL_DURATION}s)...")
		print(f"Starting continuous recording for gesture '{gesture_name}' ({TOTAL_DURATION}s total)...")
		ser.reset_input_buffer()
		overall_start = time.time()

		last_plot_time = 0
		current_rep = 1
		print(f"Recording repetition {current_rep} of {REPETITIONS} ({DURATION_PER_REP}s)...")

		# Loop until the total duration for this gesture elapses
		while (time.time() - overall_start) < TOTAL_DURATION:
			try:
				if ser.in_waiting:
					raw_line = ser.readline().decode(errors="ignore").strip()
					if not raw_line:
						continue
					ch1, ch2 = parse_dual_channel(raw_line)
					if ch1 is None or ch2 is None:
						# Could not parse this line; skip
						continue

					t = time.time() - overall_start
					# Determine which repetition this timestamp belongs to (1-based)
					rep_idx = int(t // DURATION_PER_REP) + 1
					if rep_idx > REPETITIONS:
						rep_idx = REPETITIONS

					# If we've entered a new repetition, notify the user
					if rep_idx != current_rep:
						current_rep = rep_idx
						print(f"Recording repetition {current_rep} of {REPETITIONS} ({DURATION_PER_REP}s)...")

					timestamps.append(t)
					emg1.append(ch1)
					emg2.append(ch2)

					ms = int((t) * 1000)
					rows.append([ms, ch1, adc_to_volt(ch1), ch2, adc_to_volt(ch2), rep_idx])

				# Update plot at a limited rate (PLOT_UPDATE_INTERVAL)
				now = time.time()
				if now - last_plot_time >= PLOT_UPDATE_INTERVAL:
					xs = np.array(timestamps)
					ys1 = np.array(emg1)
					ys2 = np.array(emg2)

					if xs.size > 0:
						# For performance, we display all collected points but keep axes limits fixed on x
						line1.set_data(xs, ys1)
						line2.set_data(xs, ys2)

						# Autoscale y independently
						ax1.relim(); ax1.autoscale_view()
						ax2.relim(); ax2.autoscale_view()

					# Draw
					fig.canvas.draw()
					fig.canvas.flush_events()
					last_plot_time = now

				# small sleep to yield
				time.sleep(0.001)

			except Exception as e:
				print("⚠️ Error while reading/parsing serial data:", e)
				# continue collecting if possible

		print("Done. Continuous recording finished for this gesture.")
		print("Recording complete! Saving all data...")
		ensure_dirs()

		# Save CSV for the gesture
		if len(rows) == 0:
			print("⚠️ No data was captured for this gesture.")
		else:
			df = pd.DataFrame(rows, columns=["ms", "adc_ch1", "volt_ch1", "adc_ch2", "volt_ch2", "rep"])
			save_path = os.path.join(SAVE_DIR, f"{gesture_name}.csv")
			df.to_csv(save_path, index=False)
			print(f"Saving data to: {save_path}")

			# Save the final plot image (showing entire collected duration)
			# Recompute limits and draw a final high-quality saved image
			if len(timestamps) > 0:
				xs = np.array(timestamps)
				ys1 = np.array(emg1)
				ys2 = np.array(emg2)
				line1.set_data(xs, ys1)
				line2.set_data(xs, ys2)
				ax1.relim(); ax1.autoscale_view()
				ax2.relim(); ax2.autoscale_view()
				ax2.set_xlim(0, TOTAL_DURATION)
				fig.tight_layout(rect=[0, 0.03, 1, 0.95])
				graph_path = os.path.join(GRAPH_DIR, f"{gesture_name}.png")
				fig.savefig(graph_path, dpi=200)
				print(f"Saved plot to: {graph_path}")

	finally:
		plt.ioff()
		plt.close(fig)


def main():
	print("🔌 Connecting to ESP32...")
	try:
		ser = serial.Serial(PORT, BAUD, timeout=0.1)
		# Small delay to let the serial connection initialize
		time.sleep(2)
		print(f"✅ Connected to {PORT} at {BAUD} baud.")
	except Exception as e:
		print(f"❌ Could not open serial port {PORT}: {e}")
		return

	try:
		for gesture in GESTURES:
			record_gesture(ser, gesture)

		print("\n🎉 All recordings complete! Files saved in 'data_raw/' and graphs in 'graphs/'.\n")

	finally:
		try:
			ser.close()
		except Exception:
			pass


if __name__ == "__main__":
	main()

