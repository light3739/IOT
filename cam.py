import json
from datetime import datetime

import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import deque
import logging
import argparse


class ObjectDetector:
    def __init__(self, mqtt_client, model_path='yolov8n.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.fps_buffer = deque(maxlen=30)
        self.mqtt_client = mqtt_client
        self.object_topic = "detected_objects"
        self.colors = {}

        # Добавляем переменные для контроля отправки
        self.last_mqtt_send = 0
        self.mqtt_send_interval = 5.0  # Интервал отправки в секундах
        self.last_detected_objects = {}  # Сохраняем последние обнаруженные объекты

    def get_color(self, cls_num):
        """Получение цвета из кэша или создание нового"""
        if cls_num not in self.colors:
            self.colors[cls_num] = tuple(np.random.randint(0, 255, size=3).tolist())
        return self.colors[cls_num]

    def process_frame(self, frame):
        """Обработка кадра и отправка данных в MQTT"""
        start_time = time.time()

        try:
            # Предобработка изображения
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Детекция объектов
            results = self.model(processed_frame, stream=True)

            # Обработка результатов и отправка в MQTT
            detected_objects = self.process_detections(frame, results)

            # Расчет FPS
            fps = 1.0 / max(time.time() - start_time, 0.001)
            self.fps_buffer.append(fps)

            return frame, np.mean(self.fps_buffer), detected_objects

        except Exception as e:
            logging.error(f"Error in process_frame: {e}")
            return frame, 0, {}

    def process_detections(self, frame, results):
        """Обработка детекций и отправка в MQTT"""
        current_detected_objects = {}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]

                # Подсчет объектов
                current_detected_objects[class_name] = current_detected_objects.get(class_name, 0) + 1

                # Отрисовка на кадре
                color = self.get_color(cls)
                self.draw_detection(frame, x1, y1, x2, y2, class_name, confidence, color)

        # Проверяем время и изменения объектов перед отправкой
        current_time = time.time()
        if (current_time - self.last_mqtt_send >= self.mqtt_send_interval or
                current_detected_objects != self.last_detected_objects):

            if current_detected_objects:  # Отправляем только если есть объекты
                message = {
                    "timestamp": datetime.now().isoformat(),
                    "objects": current_detected_objects,
                    "fps": float(np.mean(self.fps_buffer)) if len(self.fps_buffer) > 0 else 0.0
                }
                try:
                    self.mqtt_client.publish(
                        self.object_topic,
                        json.dumps(message),
                        qos=0
                    )
                    logging.info(f"Published objects: {current_detected_objects}")
                    self.last_mqtt_send = current_time
                    self.last_detected_objects = current_detected_objects.copy()
                except Exception as e:
                    logging.error(f"Failed to publish to MQTT: {e}")

        # Отрисовка статистики
        self.draw_stats(frame, current_detected_objects)

        return current_detected_objects

    def draw_detection(self, frame, x1, y1, x2, y2, class_name, confidence, color):
        """Отрисовка одной детекции"""
        # Рамка
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Текст с фоном
        label = f"{class_name}: {confidence:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def draw_stats(self, frame, detected_objects):
        """Отображение статистики на кадре"""
        y_offset = 30
        # FPS
        fps_text = f"FPS: {np.mean(self.fps_buffer):.1f}"
        cv2.putText(frame, fps_text,
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Список объектов
        for obj_name, count in detected_objects.items():
            y_offset += 30
            cv2.putText(frame, f"{obj_name}: {count}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
