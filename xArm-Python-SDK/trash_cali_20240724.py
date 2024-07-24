# S/N : XYZARIS0V3P2311N03
# Robot IP : 192.168.1.167
# code_version : 3.1.5.2


#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2022, UFACTORY, Inc.
# All rights reserved.
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
# Notice
#   1. Changes to this file on Studio will not be preserved
#   2. The next conversion will overwrite the file with the same name
#
# xArm-Python-SDK: https://github.com/xArm-Developer/xArm-Python-SDK
#   1. git clone git@github.com:xArm-Developer/xArm-Python-SDK.git
#   2. cd xArm-Python-SDK
#   3. python setup.py install
"""
import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI

from threading import Thread, Event
import socket
import json
import os

import threading

from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial.distance import cdist

import logging


class YOLOMain:
    def __init__(self, robot_main):
        # 모델 로드
        self.model = YOLO('/home/beakhongha/collision avoidance/train18/weights/best.pt')

        # 캘리브레이션 데이터 로드
        self.detection_duration = 0
        # 변수 초기화
        self.center_x_mm = None
        self.center_y_mm = None
        self.last_cup_center = None
        self.stable_duration = 2 

        # 카메라 열기
        self.webcam = cv2.VideoCapture(2)  # 웹캠 장치 열기
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 프레임 너비 설정
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 프레임 높이 설정
        self.robot = robot_main

        if not self.webcam.isOpened():  # 웹캠이 열리지 않은 경우
            print("웹캠을 열 수 없습니다. 프로그램을 종료합니다.")  # 오류 메시지 출력
            exit()  # 프로그램 종료

        self.camera_points =  np.array([
            [118, 210],  # 기준점 1의 카메라 좌표
            [114, 271],  # 기준점 2의 카메라 좌표
            [110, 333],  # 기준점 3의 카메라 좌표
            [480, 210],  # 기준점 4의 카메라 좌표
            [486, 268],  # 기준점 5의 카메라 좌표
            [490, 330],  # 기준점 6의 카메라 좌표
            [424, 267],  # 기준점 7의 카메라 좌표
            ], dtype=np.float32)
        
        self.robot_points = np.array([
            [300, -101],  # 기준점 1의 로봇 좌표
            [296, 0.1],  # 기준점 2의 로봇 좌표
            [298.6, 99.6],  # 기준점 3의 로봇 좌표
            [-295.4, -96.6],  # 기준점 4의 로봇 좌표
            [-296, 0.8],  # 기준점 5의 로봇 좌표
            [-301.5, 96.8],  # 기준점 6의 로봇 좌표
            [-198, -2.5],  # 기준점 7의 로봇 좌표
            ], dtype=np.float32)
        
        self.H = self.compute_homography_matrix()

    def compute_homography_matrix(self):
        H, _ = cv2.findHomography(self.camera_points, self.robot_points)
        print("호모그래피 변환 행렬 H:\n", H)
        return H
    
    def update_coordinates(self, center_x_mm, center_y_mm):
        self.robot.set_center_coordinates(center_x_mm,center_y_mm)
                                          
    def transform_to_robot_coordinates(self, image_points):
        camera_coords = np.array([image_points], dtype=np.float32)
        camera_coords = np.array([camera_coords])
        robot_coords = cv2.perspectiveTransform(camera_coords, self.H)
        # 좌표를 소수점 한 자리로 반올림
        robot_coords = [round(float(coord), 1) for coord in robot_coords[0][0]]
        return robot_coords

    def predict_on_image(self, img, conf):
        result = self.model(img, conf=conf)[0]

        # Detection
        cls = result.boxes.cls.cpu().numpy() if result.boxes else []  # 클래스, (N, 1)
        probs = result.boxes.conf.cpu().numpy() if result.boxes else []  # 신뢰도 점수, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes else []   # 박스 좌표, xyxy 형식, (N, 4)

        # Segmentation
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []  # 마스크, (N, H, W)
        
        return boxes, masks, cls, probs  # 예측 결과 반환

    def overlay(self, image, mask, color, alpha=0.5):
        """이미지와 세그멘테이션 마스크를 결합하여 하나의 이미지를 만듭니다."""
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # 마스크를 이미지 크기로 리사이즈
        colored_mask = np.zeros_like(image, dtype=np.uint8)  # 이미지와 같은 크기의 색 마스크 생성
        for c in range(3):  # BGR 각 채널에 대해
            colored_mask[:, :, c] = mask * color[c]  # 마스크를 색상으로 칠함
        
        mask_indices = mask > 0  # 마스크가 적용된 부분의 인덱스
        if mask_indices.any():  # mask_indices가 유효한지 확인
            overlay_image = image.copy()  # 원본 이미지를 복사하여 오버레이 이미지 생성
            overlay_image[mask_indices] = cv2.addWeighted(image[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)  # 마스크 부분만 밝기 조절
            return overlay_image  # 오버레이된 이미지 반환
        else:
            return image  # 유효하지 않으면 원본 이미지 반환

    def find_contours(self, mask):
        """마스크에서 외곽선을 찾습니다."""
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def distance_between_points(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def segmentation(self):

        global A_ZONE, B_ZONE, C_ZONE, NOT_SEAL                         # ROI 내에서 capsule/capsule_not_label 객체가 인식되었는지 여부
        global A_ZONE_start_time, B_ZONE_start_time, C_ZONE_start_time  # ROI 내에서 capsule 객체가 몇 초 동안 인식되었는지 여부 확인

        A_ZONE, B_ZONE, C_ZONE, NOT_SEAL = False, False, False, False
        A_ZONE_start_time, B_ZONE_start_time, C_ZONE_start_time = None, None, None
        self.start_time = None
        self.robot.cup_trash_detected = False

        # YOLO 모델의 로깅 레벨 설정
        logging.getLogger('ultralytics').setLevel(logging.ERROR)

        # 라벨별 색상 정의 (BGR 형식)
        colors = {
            'cup': (0, 255, 0),  # 컵: 녹색
            'capsule': (0, 0, 255),  # 캡슐: 빨간색
            'capsule_label': (255, 255, 0),  # 캡슐 라벨: 노란색
            'capsule_not_label': (0, 255, 255),  # 캡슐 비라벨: 청록색
            'robot': (0, 165, 255),  # 로봇: 오렌지색
            'human': (255, 0, 0),  # 인간: 파란색
            'hand': (0, 255, 255)  # 손: 노란색
        }

        # 영구적으로 설정된 ROI 구역
        rois = [(455, 65, 95, 95), (360, 65, 95, 95), (265, 65, 95, 95)]  # A_ZONE, B_ZONE, C_ZONE 순서
        specific_roi = (450, 230, 110, 110)  # Seal check ROI 구역

        while True:
            ret, frame = self.webcam.read()  # 웹캠에서 프레임 읽기
            if not ret:  # 프레임을 읽지 못한 경우
                print("카메라에서 프레임을 읽을 수 없습니다. 프로그램을 종료합니다.")  # 오류 메시지 출력
                break  # 루프 종료

            # 현재 프레임 예측
            boxes, masks, cls, probs = self.predict_on_image(frame, conf=0.5)

            # 원본 이미지에 마스크 오버레이 및 디텍션 박스 표시
            image_with_masks = np.copy(frame)  # 원본 이미지 복사

            robot_contours = []
            human_contours = []

            for box, mask, class_id, prob in zip(boxes, masks, cls, probs):  # 각 객체에 대해
                label = self.model.names[int(class_id)]  # 클래스 라벨 가져오기

                if label == 'hand':  # 'hand' 객체를 'human' 객체로 변경
                    label = 'human'

                color = colors.get(label, (255, 255, 255))  # 클래스에 해당하는 색상 가져오기
                
                if mask is not None and len(mask) > 0:
                    # 마스크 오버레이
                    image_with_masks = self.overlay(image_with_masks, mask, color, alpha=0.3)

                    # 라벨별 외곽선 저장
                    contours = self.find_contours(mask)
                    if label == 'robot':
                        robot_contours.extend(contours)
                    elif label == 'human':
                        human_contours.extend(contours)

                # 디텍션 박스 및 라벨 표시
                x1, y1, x2, y2 = map(int, box)  # 박스 좌표 정수형으로 변환
                cv2.rectangle(image_with_masks, (x1, y1), (x2, y2), color, 2)  # 경계 상자 그리기

                # trash mode에 사용
                if label == 'capsule':  # 'capsule' 객체에 대해서만 중심 좌표 계산 및 출력
                    # center 좌표(pixel)
                    center_x_pixel = (x2 - x1) / 2 + x1
                    center_y_pixel = (y2 - y1) / 2 + y1

                    # 이미지 좌표로 실세계 좌표 계산
                    image_points = [center_x_pixel, center_y_pixel]
                    world_points = self.transform_to_robot_coordinates(image_points)
                    # print(world_points, "::", image_points)
                    self.center_x_mm = world_points[0]
                    self.center_y_mm = world_points[1]

                    # print(f"center point : ({self.center_x_mm:.3f}, {self.center_y_mm:.3f})")
                    cv2.putText(image_with_masks, f'Center: ({int(self.center_x_mm)}, {int(self.center_y_mm)})', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 캡슐 중심 좌표 표시
                    self.update_coordinates(self.center_x_mm, self.center_y_mm)

                    # ROI 영역 내에 있는지 확인
                    roi_x1, roi_y1 = -400.0, -170.0
                    roi_x2, roi_y2 = 400.0, 145.0

                    # current_time = time.time()
                    
                    # if self.last_cup_center is not None:
                    #     if (roi_x1 <= self.center_x_mm <= roi_x2 and roi_y1 <= self.center_y_mm <= roi_y2 and 
                    #         self.distance_between_points((self.center_x_mm, self.center_y_mm), self.last_cup_center) < 10 and 
                    #         self.last_detection_time is not None and 
                    #         (current_time - self.last_detection_time) >= self.stable_duration):
                    #         self.cup_trash_detected = True
                    #     else:
                    #         self.last_detection_time = current_time
                    # else:
                    #     self.cup_trash_detected = False
                    # print(self.cup_trash_detected)

                    # self.last_cup_center = (self.center_x_mm, self.center_y_mm)
                    # self.last_detection_time = current_time

                    # 초기화 코드
                    self.detection_duration = 0

                    # 캡슐 인식 코드
                    current_time = time.time()
                                        
                    if self.last_cup_center is not None:
                        if (self.center_x_mm is not None and self.center_y_mm is not None and 
                            roi_x1 <= self.center_x_mm <= roi_x2 and roi_y1 <= self.center_y_mm <= roi_y2 and 
                            self.distance_between_points((self.center_x_mm, self.center_y_mm), self.last_cup_center) < 10):



                            if self.start_time is None and self.robot.MODE == "icecreaming":
                                print("1")
                                self.start_time = current_time

                            elif self.start_time is None:
                                print("2")
                                pass

                            elif current_time - self.start_time >= self.stable_duration :
                                print("3")
                                self.robot.cup_trash_detected = True
                                # print(f"Capsule detected for {current_time - self.start_time:.2f} seconds")
                                self.start_time = None
                                # self.robot.MODE == "icecreaming"
                            
                            # else:
                            #     # 인식된 시간을 초 단위로 출력
                            #     print(f"Capsule detected for {current_time - self.start_time:.2f} seconds")
                        else:
                            test = (self.distance_between_points((self.center_x_mm, self.center_y_mm), self.last_cup_center) < 10)
                            # print(f'조건 충족 안함1 {self.center_x_mm is not None} and {self.center_y_mm is not None} and {roi_x1 <= self.center_x_mm <= roi_x2} and {roi_y1 <= self.center_y_mm <= roi_y2} and {test}')
                            # print(roi_x1, self.center_x_mm, roi_x2)
                    else:
                        self.detection_duration = 0  # 인식되지 않으면 시간 초기화
                        # print('조건 충족 안함2')

                    self.last_cup_center = (self.center_x_mm, self.center_y_mm)
                    # self.last_detection_time = current_time
                        
                cv2.putText(image_with_masks, f'{label} {prob:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # 라벨 및 신뢰도 점수 표시

                # ROI 내 capsule 객체 인식 확인
                if label == 'capsule':  # capsule 객체만 확인
                    for i, (rx, ry, rw, rh) in enumerate(rois[:3]):  # 최대 세 개의 ROI만 확인
                        # ROI와 바운딩 박스의 교차 영역 계산
                        intersection_x1 = max(x1, rx)  # 교차 영역의 왼쪽 위 x 좌표
                        intersection_y1 = max(y1, ry)  # 교차 영역의 왼쪽 위 y 좌표
                        intersection_x2 = min(x2, rx + rw)  # 교차 영역의 오른쪽 아래 x 좌표
                        intersection_y2 = min(y2, ry + rh)  # 교차 영역의 오른쪽 아래 y 좌표

                        # 교차 영역의 면적 계산
                        intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                        # print(f'Intersection area: {intersection_area}')

                        # 바운딩 박스의 면적 계산
                        box_area = (x2 - x1) * (y2 - y1)
                        # print(f'Box area: {box_area}')

                        # 교차 영역이 바운딩 박스 면적의 80% 이상인지 여부 출력
                        is_condition_met = intersection_area >= 0.8 * box_area
                        # print(f'Condition met (intersection_area >= 0.8 * box_area): {is_condition_met}')

                        # 교차 영역이 바운딩 박스 면적의 80% 이상일 때만 True로 설정
                        if is_condition_met:
                            current_time = time.time()  # 현재 시간 기록

                            if i == 0:  # 첫 번째 ROI는 A_ZONE
                                if not A_ZONE:
                                    if A_ZONE_start_time is None:
                                        A_ZONE_start_time = current_time
                                        print('A_ZONE start time set')
                                    elif current_time - A_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 A_ZONE = True
                                        A_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - A_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    A_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신

                            elif i == 1:  # 두 번째 ROI는 B_ZONE
                                if not B_ZONE:
                                    if B_ZONE_start_time is None:
                                        B_ZONE_start_time = current_time
                                    elif current_time - B_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 B_ZONE = True
                                        B_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - B_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    B_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신

                            elif i == 2:  # 세 번째 ROI는 C_ZONE
                                if not C_ZONE:
                                    if C_ZONE_start_time is None:
                                        C_ZONE_start_time = current_time
                                    elif current_time - C_ZONE_start_time >= 2: # 2초 이상 캡슐 인식 시 C_ZONE = True
                                        C_ZONE = True
                                    else:
                                        print(f'Waiting for 2 seconds: {current_time - C_ZONE_start_time:.2f} seconds elapsed')
                                else:
                                    C_ZONE_start_time = current_time  # 상태가 이미 True인 경우, 시작 시간을 현재 시간으로 갱신

                        else:
                            if i == 0:
                                A_ZONE_start_time = None
                            elif i == 1:
                                B_ZONE_start_time = None
                            elif i == 2:
                                C_ZONE_start_time = None

                # 필요에 따라 A_ZONE, B_ZONE, C_ZONE 상태 업데이트
                # print(f"A_ZONE: {A_ZONE}, B_ZONE: {B_ZONE}, C_ZONE: {C_ZONE}")

                # 특정 ROI 내 capsule_not_label 객체 인식 확인
                if label == 'capsule_not_label':
                    rx, ry, rw, rh = specific_roi
                    # 특정 ROI와 바운딩 박스의 교차 영역 계산
                    intersection_x1 = max(x1, rx)
                    intersection_y1 = max(y1, ry)
                    intersection_x2 = min(x2, rx + rw)
                    intersection_y2 = min(y2, ry + rh)

                    # 교차 영역의 면적 계산
                    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)

                    # 바운딩 박스의 면적 계산
                    box_area = (x2 - x1) * (y2 - y1)

                    # 교차 영역이 바운딩 박스 면적의 80% 이상일 때만 True로 설정
                    if intersection_area >= 0.8 * box_area:
                        NOT_SEAL = True

            # 최단 거리 계산 및 시각화
            if robot_contours and human_contours:
                robot_points = np.vstack(robot_contours).squeeze()
                human_points = np.vstack(human_contours).squeeze()
                dists = cdist(robot_points, human_points)
                min_dist_idx = np.unravel_index(np.argmin(dists), dists.shape)
                robot_point = robot_points[min_dist_idx[0]]
                human_point = human_points[min_dist_idx[1]]
                min_distance = dists[min_dist_idx]
                min_distance_bool = True

                # 최단 거리 표시
                cv2.line(image_with_masks, tuple(robot_point), tuple(human_point), (255, 255, 255), 2)
                mid_point = ((robot_point[0] + human_point[0]) // 2, (robot_point[1] + human_point[1]) // 2)
                cv2.putText(image_with_masks, f'{min_distance:.2f}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                min_distance = 300
                min_distance_bool = False

            # 거리 조건 체크 및 로봇 일시정지 제어
            if min_distance <= 50 and min_distance_bool and self.robot.pressing == False:
                robot_state = 'robot stop'
                self.robot._arm.set_state(3)
            elif min_distance > 50 or not min_distance_bool:
                robot_state = 'robot move'
                self.robot._arm.set_state(0)

            # 설정된 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            for (x, y, w, h) in rois:
                cv2.rectangle(image_with_masks, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 각 ROI를 흰색 사각형으로 그림
            # 특정 ROI를 흰색 바운딩 박스로 그리고 선을 얇게 설정
            cv2.rectangle(image_with_masks, (specific_roi[0], specific_roi[1]), 
                          (specific_roi[0] + specific_roi[2], specific_roi[1] + specific_roi[3]), 
                          (255, 255, 255), 1)  # 특정 ROI를 흰색 사각형으로 그림

            # 화면 왼쪽 위에 최단 거리 및 로봇 상태 표시
            cv2.putText(image_with_masks, f'Distance: {min_distance:.2f}, state: {robot_state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 화면 왼쪽 위에 ROI 상태 표시
            cv2.putText(image_with_masks, f'A_ZONE: {A_ZONE}, B_ZONE: {B_ZONE}, C_ZONE: {C_ZONE}, NOT_SEAL: {NOT_SEAL}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 마스크가 적용된 프레임 표시
            cv2.imshow("Webcam with Segmentation Masks and Detection Boxes", image_with_masks)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 자원 해제
        self.webcam.release()  # 웹캠 장치 해제
        cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기


class RobotMain(object):
    """Robot Main Class"""

    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 20
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()
        self.state = 'stopped'
        self.MODE = "ready"
        self.pressing = False
        self.center_x_mm = None
        self.center_y_mm = None

        self.position_home = [179.2, -42.1, 7.4, 186.7, 41.5, -1.6] #angle
        self.position_jig_A_grab = [-257.3, -138.3, 198, 68.3, 86.1, -47.0] #linear
        self.position_jig_B_grab = [-152.3, -129.0, 198, 4.8, 89.0, -90.7] #linear
        self.position_jig_C_grab = [-76.6, -144.6, 198, 5.7, 88.9, -50.1] #linear
        self.position_sealing_check = [-136.8, 71.5, 307.6, 69.6, -73.9, -59] #Linear
        self.position_capsule_place = [234.9, 135.9, 465.9, 133.6, 87.2, -142.1] #Linear
        self.position_before_capsule_place = self.position_capsule_place.copy()
        self.position_before_capsule_place[2] += 25
        self.position_cup_grab = [214.0, -100.2, 145.0, -25.6, -88.5, 95.8] #linear
        self.position_topping_A = [-200.3, 162.8, 359.9, -31.7, 87.8, 96.1] #Linear
        self.position_topping_B = [106.5, -39.7, 15.0, 158.7, 40.4, 16.9] #Angle
        self.position_topping_C = [43.6, 137.9, 350.1, -92.8, 87.5, 5.3] #Linear
        self.position_icecream_with_topping = [168.7, 175.6, 359.5, 43.9, 88.3, 83.3] #Linear
        self.position_icecream_no_topping = [48.4, -13.8, 36.3, 193.6, 42.0, -9.2] #angle
        self.position_jig_A_serve = [-258.7, -136.4, 208.2, 43.4, 88.7, -72.2] #Linear
        self.position_jig_B_serve = [-166.8, -126.5, 200.9, -45.2, 89.2, -133.6] #Linear
        self.position_jig_C_serve = [-63.1, -138.2, 199.5, -45.5, 88.1, -112.1] #Linear
        self.position_capsule_grab = [234.2, 129.8, 464.5, -153.7, 87.3, -68.7] #Linear

    def set_center_coordinates(self, x_mm, y_mm):
        # 좌표 값을 업데이트
        self.center_x_mm = x_mm
        self.center_y_mm = y_mm
        # print(f"RobotMain received coordinates: ({self.center_x_mm}, {self.center_y_mm})")

        # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code,
                                                                                                 self._arm.connected,
                                                                                                 self._arm.state,
                                                                                                 self._arm.error_code,
                                                                                                 ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                       ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    def position_reverse_sealing_fail(self, linear_jig_position = [-257.3, -138.3, 192.1, 68.3, 86.1, -47.0]):
        reverse_position = linear_jig_position.copy()
        reverse_position[2] = reverse_position[2] - 10
        reverse_position[3] = -reverse_position[3]
        reverse_position[4] = -reverse_position[4]
        reverse_position[5] = reverse_position[5] - 180
        return reverse_position

    def socket_connect(self):

        self.HOST = '192.168.1.167'
        self.PORT = 20002
        self.BUFSIZE = 1024
        self.ADDR = (self.HOST, self.PORT)

        # self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.clientSocket.shutdown(1)
            self.clientSocket.close()
        except:
            pass

        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self
        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # self.serverSocket.allow_reuse_address = True
        while True:
            try:
                self.serverSocket.bind(self.ADDR)
                print("bind")

                while True:
                    self.serverSocket.listen(1)
                    print(f'[LISTENING] Server is listening on robot_server')
                    time.sleep(1)
                    try:
                        while True:
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                print("socket accepted")
                                break
                            except:
                                time.sleep(1)
                                print('except')
                                # break

                        break

                    except socket.timeout:
                        print("socket timeout")

                    except:
                        pass
                break
            except:
                pass
        # self.clientSocket.settimeout(10.0)
        print("accept")
        print("--client info--")
        # print(self.clientSocket)

        self.connected = True
        self.state = 'ready'

        # ------------------- receive msg start -----------
        while self.connected:
            print('loop start')
            time.sleep(0.5)
            try:
                print('waiting')
                self.clientSocket.settimeout(10.0)
                self.recv_msg = self.clientSocket.recv(1024).decode('utf-8')
                # try:
                #    self.recv_msg = self.clientSocket.recv(1024).decode('utf-8')
                # except Exception as e:
                #    self.pprint('MainException: {}'.format(e))
                print('\n' + self.recv_msg)
                if self.recv_msg == '':
                    print('here')
                    # continue
                    # pass
                    # break
                    raise Exception('empty msg')
                self.recv_msg = self.recv_msg.split('/')

                if self.recv_msg[0] == 'app_ping':
                    # print('app_ping received')
                    send_msg = 'robot_ping'
                    now_temp = arm.temperatures
                    now_cur = arm.currents
                    send_msg = [
                        {
                            'type': 'A', 'joint_name': 'Base', 'temperature': now_temp[0],
                            'current': round(now_cur[0], 3) * 100
                        }, {
                            'type': 'B', 'joint_name': 'Shoulder', 'temperature': now_temp[1],
                            'current': round(now_cur[1], 3) * 100
                        }, {
                            'type': 'C', 'joint_name': 'Elbow', 'temperature': now_temp[2],
                            'current': round(now_cur[2], 3) * 100
                        }, {
                            'type': 'D', 'joint_name': 'Wrist1', 'temperature': now_temp[3],
                            'current': round(now_cur[3], 3) * 100
                        }, {
                            'type': 'E', 'joint_name': 'Wrist2', 'temperature': now_temp[4],
                            'current': round(now_cur[4], 3) * 100
                        }, {
                            'type': 'F', 'joint_name': 'Wrist3', 'temperature': now_temp[5],
                            'current': round(now_cur[5], 3) * 100
                        }
                    ]
                    try:
                        time.sleep(0.5)
                        self.clientSocket.send(f'{send_msg}'.encode('utf-8'))
                        print('robot_ping')

                    except Exception as e:
                        self.pprint('MainException: {}'.format(e))
                        print('ping send fail')
                    # send_msg = arm.temperatures
                    if self.state == 'ready':
                        print('STATE : ready for new msg')
                    else:
                        print('STATE : now moving')
                else:
                    self.recv_msg[0] = self.recv_msg[0].replace("app_ping", "")
                    if self.recv_msg[0] in ['breath', 'greet', 'farewell' 'dance_random', 'dance_a', 'dance_b',
                                            'dance_c',
                                            'sleep', 'comeon']:
                        print(f'got message : {self.recv_msg[0]}')
                        if self.state == 'ready':
                            self.state = self.recv_msg[0]
                    elif self.recv_msg[0] == 'robot_script_stop':
                        code = self._arm.set_state(4)
                        if not self._check_code(code, 'set_state'):
                            return
                        sys.exit()
                        self.is_alive = False
                        print('program exit')

                    # 픽업존 아이스크림 뺐는지 여부 확인
                    elif self.recv_msg[0].find('icecream_go') >= 0 or self.recv_msg[0].find(
                            'icecream_stop') >= 0 and self.state == 'icecreaming':
                        print(self.recv_msg[0])
                        if self.recv_msg[0].find('icecream_go') >= 0:
                            self.order_msg['makeReq']['latency'] = 'go'
                        else:
                            self.order_msg['makeReq']['latency'] = 'stop'
                            print('000000000000000000000000000000')

                    # 실링 존재 여부 확인

                    if self.recv_msg[0].find('sealing_pass') >= 0 and self.state == 'icecreaming':
                        self.order_msg['makeReq']['sealing'] = 'go'
                        print('socket_go')
                    elif self.recv_msg[0].find('sealing_reject') >= 0 and self.state == 'icecreaming':
                        self.order_msg['makeReq']['sealing'] = 'stop'
                        print('socket_stop')

                    else:
                        # print('else')
                        try:
                            self.order_msg = json.loads(self.recv_msg[0])
                            if self.order_msg['type'] == 'ICECREAM':
                                if self.state == 'ready':
                                    print('STATE : icecreaming')
                                    print(f'Order message : {self.order_msg}')
                                    self.state = 'icecreaming'
                            # else:
                            #    self.clientSocket.send('ERROR : already moving'.encode('utf-8'))
                            else:
                                self.clientSocket.send('ERROR : wrong msg received'.encode('utf-8'))
                        except:
                            pass
                self.recv_msg[0] = 'zzz'

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
                # if e == 'empty msg' :
                #    pass
                # self.connected = False
                print('connection lost')
                while True:
                    time.sleep(2)
                    try:

                        try:
                            self.serverSocket.shutdown(socket.SHUT_RDWR)
                            self.serverSocket.close()
                        except:
                            pass

                        print('socket_making')
                        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        self.serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

                        self.serverSocket.bind(self.ADDR)
                        print("bind")

                        while True:
                            print('listening')
                            self.serverSocket.listen(1)
                            print(f'reconnecting')
                            try:
                                self.clientSocket, addr_info = self.serverSocket.accept()
                                break

                            except socket.timeout:
                                print('socket.timeout')
                                break

                            except:
                                pass
                        break
                    except Exception as e:
                        self.pprint('MainException: {}'.format(e))
                        print('except')
                        # pass

    # =================================  motion  =======================================

    def motion_dance_a(self):  # designed 'poke'
        try:
            self.clientSocket.send('dance_a_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 60
        self._angle_acc = 300
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(3)):
            if not self.is_alive:
                break
            code = self._arm.set_servo_angle(angle=[212.0, -21.0, 112.0, 207.0, -0.8, 7.3], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[212.0, -38.0, 100.3, 180.4, -6.4, 6.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        '''
        code = self._arm.set_servo_angle(angle=[329.0, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(3)):
            if not self.is_alive:
                break
            code = self._arm.set_servo_angle(angle=[329.0, -21.0, 112.0, 207.0, -0.8, 7.3], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[329.0, -38.0, 100.3, 180.4, -6.4, 6.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        '''
        self._angle_speed = 60
        self._angle_acc = 200
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return

    def motion_dance_b(self):  # designed 'shake'
        try:
            self.clientSocket.send('dance_b_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 70
        self._angle_acc = 200
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(4)):
            if not self.is_alive:
                break
            code = self._arm.set_servo_angle(angle=[220.7, -39.1, 67.0, 268.3, -40.0, -91.8], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[183.0, -39.1, 102.7, 220.0, -11.6, -140.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return

    def motion_dance_c(self):  # designed '빙글빙글'
        try:
            self.clientSocket.send('dance_c_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 150
        self._angle_acc = 700
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(3)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_servo_angle(angle=[180.0, 70.0, 250.0, 173.1, 0.0, -135.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, -70.0, 110.0, 180.0, 0.0, 135.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_servo_angle(angle=[180.0, 70.0, 250.0, 173.1, 0.0, -135.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        while True:
            try:
                self.clientSocket.send('dance_c_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def motion_come_on(self):  # designed '컴온컴온
        try:
            self.clientSocket.send('comeon_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 80
        self._angle_acc = 400
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[180.0, 70.0, 220.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=40.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        for i in range(int(2)):
            if not self.is_alive:
                break
            t1 = time.monotonic()
            code = self._arm.set_servo_angle(angle=[180.0, 70.0, 220.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 62.0, 222.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 55.0, 222.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 45.0, 222.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 35.0, 224.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 25.0, 224.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 15.0, 226.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 5.0, 226.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 0.0, 228.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 5.0, 230.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 20.0, 226.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 35.0, 226.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 45.0, 228.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 55.0, 226.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 65.0, 224.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_servo_angle(angle=[180.0, 70.0, 222.0, 90.0, 20.0, 0.0], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            interval = time.monotonic() - t1
            if interval < 0.01:
                time.sleep(0.01 - interval)
        code = self._arm.set_servo_angle(angle=[180.0, 65.0, 222.0, 90.0, 60.0, 0.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        while True:
            try:
                self.clientSocket.send('comeon_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def motion_greet(self):
        try:
            self.clientSocket.send('greet_start'.encode('utf-8'))
        except:
            print('socket error')

        self._angle_speed = 100
        self._angle_acc = 350

        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 181.5, -1.9, -92.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 180.9, -28.3, -92.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 185.4, 30.8, -94.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        try:
            self.clientSocket.send('motion_greet finish'.encode('utf-8'))
        except:
            print('socket error')
        code = self._arm.set_servo_angle(angle=[178.9, -0.7, 179.9, 181.5, -1.9, -92.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        while True:
            try:
                self.clientSocket.send('motion_greet_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def motion_breath(self):
        pass

    def motion_sleep(self):  # designed 'sleep'
        try:
            self.clientSocket.send('sleep_start'.encode('utf-8'))
        except:
            print('socket error')

        for i in range(int(1)):
            if not self.is_alive:
                break
            for i in range(int(2)):
                if not self.is_alive:
                    break
                self._angle_speed = 20
                self._angle_acc = 200
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
                self._angle_speed = 5
                self._angle_acc = 5
                code = self._arm.set_servo_angle(angle=[179.0, -10.2, 24.0, 178.2, 39.2, -2.0], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
            self._angle_speed = 30
            self._angle_acc = 300
            code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            for i in range(int(3)):
                if not self.is_alive:
                    break
                self._angle_speed = 180
                self._angle_acc = 1000
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 199.8, 43.4, -11.0],
                                                 speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
                code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 157.3, 43.2, 12.7], speed=self._angle_speed,
                                                 mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'):
                    return
            self._angle_speed = 20
            self._angle_acc = 200
            code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            code = self._arm.set_pause_time(2)
            if not self._check_code(code, 'set_pause_time'):
                return
        while True:
            try:
                self.clientSocket.send('sleep_finish'.encode('utf-8'))
                break
            except:
                print('socket error')

    def motion_clean_mode(self):
        pass

    def pin_off(self):
        self.clientSocket.send('pin_off_start'.encode('utf-8'))
        # cup_dispenser_up
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        # press_up
        code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        self.clientSocket.send('pin_off_finish'.encode('utf-8'))

    def pin_test(self):
        time.sleep(3)
        code = self._arm.set_servo_angle(angle=[179.0, -17.7, 29.0, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        code = self._arm.set_cgpio_digital(0, 1, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        time.sleep(2)
        code = self._arm.set_servo_angle(angle=[179.0, -17.7, 83.3, 177.8, 43.8, -1.4], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        time.sleep(1)
        code = self._arm.set_cgpio_digital(0, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return
        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        time.sleep(3)
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        time.sleep(3)
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

    # Robot Main Run
    def run(self):
        try:
            while self.is_alive:
                # Joint Motion
                if self.state == 'icecreaming':
                    # --------------icecream start--------------------
                    try:
                        self.clientSocket.send('icecream_start'.encode('utf-8'))
                    except:
                        print('socket error')
                    time.sleep(int(self.order_msg['makeReq']['latency']))
                    self.motion_home()
                    # self.check_gripper()
                    while True:
                        if self.order_msg['makeReq']['latency'] in ['go', 'stop']:
                            break
                        time.sleep(0.2)
                    if self.order_msg['makeReq']['latency'] in ['go']:
                        self.motion_grab_capsule()
                        if self.order_msg['makeReq']['sealing'] in ['yes']:
                            self.motion_check_sealing()
                            try:
                                self.clientSocket.send('sealing_check'.encode('utf-8'))
                            except:
                                pass
                            count = 0
                            while True:
                                # if sealing_check request arrives or 5sec past
                                if self.order_msg['makeReq']['sealing'] in ['go', 'stop'] or count >= 5:
                                    print(self.order_msg['makeReq']['sealing'])
                                    break
                                time.sleep(0.2)
                                count += 0.2
                        if self.order_msg['makeReq']['sealing'] in ['go'] or self.order_msg['makeReq']['sealing'] not in ['yes', 'stop']:
                            #print('sealing_pass')
                            self.motion_place_capsule()
                            self.motion_grab_cup()
                            self.motion_topping()
                            self.motion_make_icecream()
                            self.motion_serve()
                            self.motion_trash_capsule()
                            self.motion_home()
                            print('icecream finish')
                            while True:
                                try:
                                    self.clientSocket.send('icecream_finish'.encode('utf-8'))
                                    break
                                except:
                                    time.sleep(0.2)
                                    print('socket_error')
                        else:
                            self.motion_place_fail_capsule()
                            self.motion_home()
                            self.clientSocket.send('icecream_cancel'.encode('utf-8'))
                            self.order_msg['makeReq']['sealing'] = ''
                    else:
                        while True:
                            try:
                                self.clientSocket.send('icecream_cancel'.encode('utf-8'))
                                break
                            except:
                                print('socket error')
                        self.order_msg['makeReq']['latency'] = 0
                    print('sendsendsendsnedasdhfaenbeijakwlbrsvz;ikbanwzis;fklnairskjf')
                    self.state = 'ready'

                elif self.state == 'test':
                    try:
                        self.clientSocket.send('test_start'.encode('utf-8'))
                    except:
                        print('socket error')
                    # self.motion_home()
                    # self.motion_grab_cup()
                    # self.motion_serve()

                elif self.state == 'greet':
                    self.motion_greet()
                    self.motion_home()
                    while True:
                        try:
                            self.clientSocket.send('greet_finish'.encode('utf-8'))
                            break
                        except:
                            print('socket error')
                            time.sleep(0.2)
                    print('greet finish')
                    self.state = 'ready'

                elif self.state == 'dance_random':
                    dance_num = random.randrange(1, 4)
                    if dance_num == 1:
                        self.motion_dance_a()
                    elif dance_num == 2:
                        self.motion_dance_b()
                    elif dance_num == 3:
                        self.motion_dance_c()
                    while True:
                        try:
                            self.clientSocket.send('dance_random_finish'.encode('utf-8'))
                            break
                        except:
                            print('socket error')
                            time.sleep(0.2)
                    self.state = 'ready'

                elif self.state == 'dance_a':
                    self.motion_dance_a()
                    self.motion_home()
                    while True:
                        try:
                            self.clientSocket.send('dance_a_finish'.encode('utf-8'))
                            break
                        except:
                            print('socket error')
                            time.sleep(0.2)
                    self.state = 'ready'

                elif self.state == 'dance_b':
                    self.motion_dance_b()
                    self.motion_home()
                    while True:
                        try:
                            self.clientSocket.send('dance_b_finish'.encode('utf-8'))
                            break
                        except:
                            print('socket error')
                            time.sleep(0.2)
                    self.state = 'ready'

                elif self.state == 'dance_c':
                    self.motion_dance_c()
                    self.motion_home()
                    # self.clientSocket.send('dance_c_finish'.encode('utf-8'))
                    self.state = 'ready'

                elif self.state == 'breath':
                    try:
                        self.clientSocket.send('breath_start'.encode('utf-8'))
                        time.sleep(5)
                        self.clientSocket.send('breath_finish'.encode('utf-8'))
                    except:
                        print('socket error')

                elif self.state == 'sleep':
                    self.motion_sleep()
                    self.motion_home()
                    while True:
                        try:
                            self.clientSocket.send('sleep_finish'.encode('utf-8'))
                            break
                        except:
                            print('socket error')
                            time.sleep(0.2)
                    self.state = 'ready'

                elif self.state == 'comeon':
                    print('come_on start')
                    self.motion_come_on()
                    # self.motion_home()
                    self.state = 'ready'

                elif self.state == 'clean_mode':
                    try:
                        self.clientSocket.send('clean_mode_start'.encode('utf-8'))
                    except:
                        print('socket error')
                    self.state = 'ready'

                    code = self._arm.set_cgpio_digital(1, 1, delay_sec=0)
                    if not self._check_code(code, 'set_cgpio_digital'):
                        return
                    self.state = 'ready'

                elif self.state == 'clean_mode_end':
                    code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
                    if not self._check_code(code, 'set_cgpio_digital'):
                        return
                    self.state = 'ready'


                elif self.state == 'ping':
                    print('ping checked')
                    # self.motion_home()
                    self.state = 'ready'

                else:
                    pass

                # self.state = 'ready'
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)

    def motion_trash_cup(self, position) :
        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        try:
            self.clientSocket.send('motion_trash_cup_start'.encode('utf-8'))
        except:
            print('socket error')

        code = self._arm.set_servo_angle(angle=[176, 31.7, 31, 76.7, 91.2, -1.9], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        if position == 'A':

            code = self._arm.set_servo_angle(angle=[179.5, 33.5, 32.7, 113.0, 93.1, -2.3], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'):
                return

            code = self._arm.set_position(*self.position_jig_A_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return

        elif position == 'B':

            code = self._arm.set_position(*self.position_jig_B_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return

        elif position == 'C':

            code = self._arm.set_servo_angle(angle=[182.6, 27.8, 27.7, 55.7, 90.4, -6.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'):
                return
            
            code = self._arm.set_position(*self.position_jig_C_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'):
                return

        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return

        time.sleep(1)

        code = self._arm.set_servo_angle(angle=[176, 31.7, 31, 76.7, 91.2, -1.9], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        code = self._arm.set_servo_angle(angle=[152.6, 11.5, 17.1, 238.1, 91.2, -1.9], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        

        # home
        code = self._arm.set_servo_angle(angle=[152.6, 11.5, 17.1, 186.7, 91.2, -1.9], speed=self._angle_speed, 
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed, 
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'):
            return
        
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return     
        
    def joint_state(self):
        while self.is_alive:
            print(f'joint temperature : {arm.temperatures}')
            time.sleep(0.5)
            print(f'joint current : {arm.currents}')
            time.sleep(10)
        




    # ============================== aris project code ==============================

    def motion_home_test(self):

        print('motion_home start')

        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

        # press_up
        code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return

        # Joint Motion
        self._angle_speed = 80
        self._angle_acc = 200

        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        print('motion_home finish')

    def motion_grab_capsule_test(self):

        print('motion_grab_capsule start')

        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        
        # Joint Motion
        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        if A_ZONE:
            pass
        else:
            code = self._arm.set_servo_angle(angle=[176, 31.7, 31, 76.7, 91.2, -1.9], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        if A_ZONE:
            code = self._arm.set_servo_angle(angle=[179.5, 33.5, 32.7, 113.0, 93.1, -2.3], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_jig_A_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_servo_angle'): return

        elif B_ZONE:
            code = self._arm.set_position(*self.position_jig_B_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        elif C_ZONE:
            code = self._arm.set_servo_angle(angle=[182.6, 27.8, 27.7, 55.7, 90.4, -6.4], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*self.position_jig_C_grab, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

        if C_ZONE:
            code = self._arm.set_position(z=150, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=False)
            if not self._check_code(code, 'set_position'): return
            
            self._tcp_speed = 200
            self._tcp_acc = 1000

            code = self._arm.set_tool_position(*[0.0, 0.0, -90.0, 0.0, 0.0, 0.0], speed=self._tcp_speed,
                                               mvacc=self._tcp_acc, wait=False)
            if not self._check_code(code, 'set_servo_angle'): return
            
        else:
            code = self._arm.set_position(z=100, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=False)
            if not self._check_code(code, 'set_position'): return
            
        self._angle_speed = 180
        self._angle_acc = 500
            
        code = self._arm.set_servo_angle(angle=[145, -18.6, 10.5, 97.5, 81.4, 145], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        print('motion_grab_capsule finish')

    def motion_check_sealing_test(self):

        print('motion_check_sealing start')

        self._angle_speed = 200
        self._angle_acc = 200

        code = self._arm.set_position(*self.position_sealing_check, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        print('motion_check_sealing finish')

    def motion_place_fail_capsule_test(self):

        print('motion_place_fail_capsule start')

        if A_ZONE:
            code = self._arm.set_servo_angle(angle=[177.3, 5.5, 12.9, 133.6, 81.3, 183.5], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_A_grab), speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

        elif B_ZONE:
            code = self._arm.set_servo_angle(angle=[159.5, 11.8, 22.2, 75.6, 92.8, 186.6], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_B_grab) , speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        elif C_ZONE:
            code = self._arm.set_servo_angle(angle=[176.9, -2.2, 15.3, 69.3, 87.5, 195.5], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=False, radius=20.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_reverse_sealing_fail(self.position_jig_C_grab) , speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        code = self._arm.set_position(z=100, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=False)
        if not self._check_code(code, 'set_position'): return
        
        print('motion_place_fail_capsule finish')

    def motion_place_capsule_test(self):

        print('motion_place_capsule start')
        
        code = self._arm.set_servo_angle(angle=[81.0, -10.8, 6.9, 103.6, 88.6, 9.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=40.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[10, -20.8, 7.1, 106.7, 79.9, 26.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return
                
        code = self._arm.set_servo_angle(angle=[8.4, -42.7, 23.7, 177.4, 31.6, 3.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=40.0)
        if not self._check_code(code, 'set_servo_angle'): return
                
        code = self._arm.set_servo_angle(angle=[8.4, -32.1, 55.1, 96.6, 29.5, 81.9], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*self.position_before_capsule_place, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
                
        code = self._arm.set_position(*self.position_capsule_place, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_cgpio_analog(0, 0)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(2)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        time.sleep(0.5)

        print('motion_place_capsule finish')
        time.sleep(0.5)

    def motion_grab_cup_test(self):

        print('motion_grab_cup start')

        code = self._arm.set_position(*[233.4, 10.3, 471.1, -172.2, 87.3, -84.5], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=20.0, wait=False)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        time.sleep(1)

        code = self._arm.set_servo_angle(angle=[-2.8, -2.5, 45.3, 119.8, -79.2, -18.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*[195.0, -96.5, 200.8, -168.0, -87.1, -110.5], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=10.0, wait=False)
        if not self._check_code(code, 'set_position'): return

        code = self._arm.set_position(*self.position_cup_grab, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(2)

        code = self._arm.set_position(z=120, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_servo_angle(angle=[2.9, -31.0, 33.2, 125.4, -30.4, -47.2], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

        print('motion_grab_cup finish')
        time.sleep(0.5)

    def motion_topping_test(self):

        global toppingAmount
        toppingAmount = 5

        print('motion_topping start')
        print('send')

        if self.Toping:
            code = self._arm.set_servo_angle(angle=[36.6, -36.7, 21.1, 85.6, 59.4, 44.5], speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            if C_ZONE:
                code = self._arm.set_position(*self.position_topping_C, speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_cgpio_digital(2, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_pause_time(toppingAmount - 3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_pause_time(2)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(2, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
                                                relative=True, wait=False)
                if not self._check_code(code, 'set_position'): return

            elif B_ZONE:
                code = self._arm.set_servo_angle(angle=[55.8, -48.2, 14.8, 86.1, 60.2, 58.7], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=20.0)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_servo_angle(angle=self.position_topping_B, speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_cgpio_digital(1, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_position(z=20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                                wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_pause_time(toppingAmount - 4)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_pause_time(3)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                code = self._arm.set_cgpio_digital(1, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc,
                                                relative=True, wait=False)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_servo_angle(angle=[87.5, -48.2, 13.5, 125.1, 44.5, 46.2], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=10.0)
                if not self._check_code(code, 'set_servo_angle'): return

                code = self._arm.set_position(*[43.6, 137.9, 350.1, -92.8, 87.5, 5.3], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return

            elif A_ZONE:
                code = self._arm.set_position(*self.position_topping_A, speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=0.0, wait=True)
                if not self._check_code(code, 'set_position'): return

                code = self._arm.set_cgpio_digital(0, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_pause_time(toppingAmount - 1)
                if not self._check_code(code, 'set_servo_angle'): return

                
                
                code = self._arm.set_pause_time(0)
                if not self._check_code(code, 'set_pause_time'):
                    return
                
                self.pressing = True
                code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return
                
                code = self._arm.set_cgpio_digital(0, 0, delay_sec=0)
                if not self._check_code(code, 'set_cgpio_digital'):
                    return

                code = self._arm.set_servo_angle(angle=[130.0, -33.1, 12.5, 194.3, 51.0, 0.0], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
                if not self._check_code(code, 'set_servo_angle'): return
                
                code = self._arm.set_position(*[-38.2, 132.2, 333.9, -112.9, 86.3, -6.6], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return
                
                code = self._arm.set_position(*[43.6, 137.9, 350.1, -92.8, 87.5, 5.3], speed=self._tcp_speed,
                                                mvacc=self._tcp_acc, radius=10.0, wait=False)
                if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*self.position_icecream_with_topping, speed=self._tcp_speed,
                                            mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        else:
            self.pressing = True
            code = self._arm.set_cgpio_digital(3, 1, delay_sec=0)
            if not self._check_code(code, 'set_cgpio_digital'):
                return
            code = self._arm.set_servo_angle(angle=self.position_icecream_no_topping, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

        print('motion_topping finish')
        time.sleep(0.5)

    def motion_make_icecream_test(self):

        print('motion_make_icecream start')

        if self.Toping:
            time.sleep(4)
        else:
            time.sleep(7)

        time.sleep(3)
        code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return

        time.sleep(3)
        code = self._arm.set_position(z=-10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        if not self._check_code(code, 'set_pause_time'):
            return

        code = self._arm.set_position(z=-50, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        time.sleep(1)
        self.pressing = False
        code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        if not self._check_code(code, 'set_cgpio_digital'):
            return

        print('motion_make_icecream finish')
        time.sleep(0.5)

    def motion_serve_test(self):

        print('motion_serve start')

        code = self._arm.set_servo_angle(angle=[18.2, -12.7, 8.3, 90.3, 88.1, 23.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=20.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[146.9, -12.7, 8.3, 91.0, 89.3, 22.1], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._tcp_speed = 100
        self._tcp_acc = 1000

        if A_ZONE:
            code = self._arm.set_position(*self.position_jig_A_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-18, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-256.2, -126.6, 210.1, -179.2, 77.2, 66.9], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)
            code = self._arm.set_position(*[-242.8, -96.3, 210.5, -179.2, 77.2, 66.9], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(*[-189.7, -26.0, 193.3, -28.1, 88.8, -146.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            

        elif B_ZONE:

            code = self._arm.set_position(*self.position_jig_B_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=False)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-13, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)
            code = self._arm.set_position(*[-165.0, -122.7, 200, -178.7, 80.7, 92.5], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)
            code = self._arm.set_position(*[-165.9, -81.9, 200, -178.7, 80.7, 92.5], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(*[-168.5, -33.2, 192.8, -92.9, 86.8, -179.3], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        elif C_ZONE:
            code = self._arm.set_servo_angle(angle=[177.6, 0.2, 13.5, 70.0, 94.9, 13.8], speed=self._angle_speed,
                                             mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_position(*self.position_jig_C_serve, speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.set_position(z=-12, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                          wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            time.sleep(1)

            code = self._arm.set_position(*[-75, -132.8, 208, -176.8, 76.1, 123.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
            time.sleep(0.5)

            code = self._arm.set_position(*[-92.0, -107.5, 208, -176.8, 76.1, 123.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*[-98.1, -52.1, 191.4, -68.4, 86.4, -135.0], speed=self._tcp_speed,
                                          mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        time.sleep(0.5)
        code = self._arm.set_servo_angle(angle=[169.6, -8.7, 13.8, 85.8, 93.7, 19.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=10.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._tcp_speed = 100
        self._tcp_acc = 1000

        print('motion_serve finish')

    def motion_trash_capsule_test(self):

        print('motion_trash_capsule start')

        self._angle_speed = 150
        self._angle_acc = 300

        code = self._arm.set_servo_angle(angle=[51.2, -8.7, 13.8, 95.0, 86.0, 17.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[-16.2, -19.3, 42.7, 82.0, 89.1, 55.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=[-19.9, -19.1, 48.7, 87.2, 98.7, 60.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_position(*[222.8, 0.9, 470.0, -153.7, 87.3, -68.7], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.set_position(*self.position_capsule_grab, speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        code = self._arm.close_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(1)

        code = self._arm.set_position(z=30, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.set_position(*[221.9, -5.5, 500.4, -153.7, 87.3, -68.7], speed=self._tcp_speed,
                                      mvacc=self._tcp_acc, radius=0.0, wait=True)
        if not self._check_code(code, 'set_position'): return
        
        self._angle_speed = 60
        self._angle_acc = 100

        code = self._arm.set_servo_angle(angle=[-10.7, -2.4, 53.5, 50.4, 78.1, 63.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=10.0)
        if not self._check_code(code, 'set_servo_angle'): return

        self._angle_speed = 160
        self._angle_acc = 1000

        code = self._arm.set_servo_angle(angle=[18.0, 11.2, 40.4, 90.4, 58.7, -148.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'open_lite6_gripper'):
            return
        # time.sleep(2)

        code = self._arm.set_servo_angle(angle=[25.2, 15.2, 42.7, 83.2, 35.0, -139.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[18.0, 11.2, 40.4, 90.4, 58.7, -148.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_servo_angle(angle=[25.2, 15.2, 42.7, 83.2, 35.0, -139.8], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'stop_lite6_gripper'):
            return
        self._angle_speed = 120
        self._angle_acc = 1000

        code = self._arm.set_servo_angle(angle=[28.3, -9.0, 12.6, 85.9, 78.5, 20.0], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=30.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[149.3, -9.4, 10.9, 114.7, 69.1, 26.1], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=False, radius=50.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[179.2, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        print('motion_trash_capsule finish')
        time.sleep(0.5)

    def robot_pause(self):
        global robot_state
        if robot_state == 'robot stop':
            self._arm.set_state(3)
        else:
            self._arm.set_state(0)


    # ==================== pause test motion ====================
    def run_robot_pause_test(self):
        self._angle_speed = 10
        self._angle_acc = 10

        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[191.9, -0.3, 26.4, 181.2, 66.8, -6.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[337.3, 4, 36.3, 179.3, 55.5, -1.4], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=[191.9, -0.3, 26.4, 181.2, 66.8, -6.6], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return

        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return


    def run_robot_test(self):

        global A_ZONE, B_ZONE, C_ZONE, NOT_SEAL

        # --------------모드 설정 변수(나중에 방식 변경)--------------

        self.Toping = True
        self.MODE = 'icecreaming'

        # --------------카메라 없이 테스트할 때 변수--------------
        A_ZONE = True
        B_ZONE = False
        C_ZONE = False
        NOT_SEAL = True

        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 1000

        print('motion_grab_cup start')
        
        code = self._arm.set_servo_angle(angle=[2.9, -31.0, 33.2, 125.4, -30.4, -47.2], speed=self._angle_speed,
                                         mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        
        code = self._arm.set_cgpio_analog(0, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return
        code = self._arm.set_cgpio_analog(1, 5)
        if not self._check_code(code, 'set_cgpio_analog'):
            return

        print('motion_grab_cup finish')
        time.sleep(0.5)

        global toppingAmount
        toppingAmount = 5

        print('motion_topping start')
        print('send')

        if self.Toping:
            code = self._arm.set_position(*self.position_icecream_with_topping, speed=self._tcp_speed,
                                            mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return
            
        else:
            code = self._arm.set_servo_angle(angle=self.position_icecream_no_topping, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

        print('motion_topping finish')
        time.sleep(0.5)

        print('motion_make_icecream start')

        if self.Toping:
            time.sleep(5)
        else:
            time.sleep(8)

        time.sleep(4)
        code = self._arm.set_position(z=-20, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return

        time.sleep(4)
        code = self._arm.set_position(z=-10, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        if not self._check_code(code, 'set_pause_time'):
            return

        code = self._arm.set_position(z=-50, radius=0, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                      wait=True)
        if not self._check_code(code, 'set_position'): return
        
        time.sleep(1)

        # code = self._arm.set_cgpio_digital(3, 0, delay_sec=0)
        # if not self._check_code(code, 'set_cgpio_digital'):
        #     return

        print('motion_make_icecream finish')
        time.sleep(0.5)



    # ============================= trash mode =============================
    def trash_check_mode(self):

        print('trash_check_mode start')

        self._angle_speed = 50
        self._angle_acc = 50

        self._tcp_speed = 50
        self._tcp_acc = 50

        # ---------- 왼쪽 탐지 ----------
        code = self._arm.set_servo_angle(angle=[180, -95, 25, 186.7, 100, -1.6], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(6)
        if self.cup_trash_detected == True:
            self.cup_trash_detected = False
            self.trash_mode()
        else:
            print(self.cup_trash_detected)
            pass
            
        self._angle_speed = 50
        self._angle_acc = 50

        self._tcp_speed = 50
        self._tcp_acc = 50
        
        # ---------- 오른쪽 탐지 ----------
        code = self._arm.set_servo_angle(angle=[180, 10, 25, 186.7, 75, -1.6], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return
        time.sleep(6)
        
        if self.cup_trash_detected == True:
            self.cup_trash_detected = False
            self.trash_mode()
        else:
            print(self.cup_trash_detected)
            pass

        
        

    def trash_mode(self):
        print('trash_mode start')

        # 테스트용 변수선언
        center_x_mm = self.center_x_mm
        center_y_mm = self.center_y_mm
        
        trash_mode_initial = [180, -27.2, 1.8, 180, 48.1, 180] #angle
        trash_mode_ReadyToTrash = [180, -35, 1, 180, 53, 180] #angle
        trash_mode_Trashing = [60, -35, 1, 180, 33, 180] #angle    
        
        self._angle_speed = 100
        self._angle_acc = 100

        self._tcp_speed = 100
        self._tcp_acc = 1000

        code = self._arm.open_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        time.sleep(0.5)
        code = self._arm.stop_lite6_gripper()
        if not self._check_code(code, 'close_lite6_gripper'):
            return
        
        code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
        if not self._check_code(code, 'set_servo_angle'): return        

        # -------------------- 쓰레기 탐지되면 동작_왼쪽 바깥쪽 --------------------
        if center_x_mm <= -300 and center_y_mm >= -130 and center_y_mm <= 100:
            code = self._arm.set_servo_angle(angle=trash_mode_initial, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(y=center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-100, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(*[center_x_mm+90, center_y_mm, 150.6, 180, -77.1, -180], speed=self._tcp_speed,
                                            mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(2)
            
            code = self._arm.set_position(z=100, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[135, 14.4, 17.3, 270.9, 83.7, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(3)

            code = self._arm.set_servo_angle(angle=[180, 14.4, 30, 275.4, 90, 162.7], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
        else:
            pass

        # -------------------- 쓰레기 탐지되면 동작_왼쪽 안쪽 --------------------
        if center_x_mm >= -300 and center_x_mm <= -100 and center_y_mm >= -130 and center_y_mm <= 110:
            code = self._arm.set_servo_angle(angle=trash_mode_initial, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_servo_angle(angle=[180, 18.4, 95.2, 180, -70.7, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return
            
            code = self._arm.set_servo_angle(angle=[180, 54.5, 117.5, 180, -77.5, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(y=center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(x=80, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[0, 0, -30, 0, -15.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, relative=True, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, 7, -1, 0, -2, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, relative=True, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            time.sleep(1)

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(3)

            code = self._arm.set_position(z=50, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                        wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[180, 36.5, 58, 180, -96.9, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[171.3, 19.3, 33.5, 131.9, -91.7, 180], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[134.5, 4.9, 14.1, 92.9, -80.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            
            time.sleep(1)
            
            code = self._arm.set_servo_angle(angle=[178.6, 4.9, 14.1, 87.5, -80.5, 0], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[178.6, -42.5, 14.1, 94.9, -87, -19], speed=self._angle_speed,
                                            mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
        else:
            pass

        # -------------------- 쓰레기 탐지되면 동작_오른쪽 --------------------
        if center_x_mm > 100 and center_x_mm < 380 and center_y_mm >= -130 and center_y_mm <= 100:
            code = self._arm.set_servo_angle(angle=[90, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=False, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, -42.1, 7.4, 186.7, 41.5, -1.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[0, 0, 28.4, 180, 64.3, 0], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=[-47.7, 6.2, 57.3, 16.5, 57.9, 31.1], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_position(*[center_x_mm, -173, 307.5, -173, 13.3, -87.6], speed=self._tcp_speed,
                                        mvacc=self._tcp_acc, radius=0.0, wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-56.2, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(z=-23, roll=27.8, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=40, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=-7, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return
        
            code = self._arm.set_position(z=-41.5, roll=27,radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_position(y=50, z=-8.5, roll=3.4,radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            if center_y_mm >= 0:
                code = self._arm.set_position(y=center_y_mm, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
                if not self._check_code(code, 'set_position'): return

            code = self._arm.close_lite6_gripper()
            if not self._check_code(code, 'close_lite6_gripper'):
                return
            
            time.sleep(2)

            code = self._arm.set_position(z=23, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True,
                                            wait=True)
            if not self._check_code(code, 'set_position'): return

            code = self._arm.set_servo_angle(angle=[22.6, 0, 14.5, 116.1, 75.6, 180], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.open_lite6_gripper()
            if not self._check_code(code, 'open_lite6_gripper'):
                return
            
            time.sleep(1)

            code = self._arm.set_servo_angle(angle=[90, -53.4, 9.5, 157.3, 21.1, 26.6], speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.set_servo_angle(angle=self.position_home, speed=self._angle_speed,
                                                    mvacc=self._angle_acc, wait=True, radius=0.0)
            if not self._check_code(code, 'set_servo_angle'): return

            code = self._arm.stop_lite6_gripper()
            if not self._check_code(code, 'stop_lite6_gripper'):
                return
        else:
            pass

        print('trash_mode finish')



    # ============================= main =============================
    def run_robot(self):

        # --------------모드 설정 변수(나중에 방식 변경)--------------
        self.Toping = True
        self.MODE = 'icecreaming'
        global A_ZONE, B_ZONE, C_ZONE
        self.A_ZONE, self.B_ZONE, self.C_ZONE = False, False, False

        while self.is_alive:
            # if self.order_list != []:
            #     self.MODE = 'icecreaming'
            #     raw_order = self.order_list.pop(0)
            #     order = raw_order

            # elif self.gritting_list != []:
            #     self.MODE = 'gritting'
            #     data = self.gritting_list.pop(0)
            #     gender = data[0]
            #     age = data[1]
            # else:
            #     self.MODE = 'ready'
            # Joint Motion
            if self.MODE == 'icecreaming':
                # --------------icecream start--------------------
                print('icecream start')
                time.sleep(4)
                self.motion_home_test()
                time.sleep(1)
                self.trash_check_mode()
                time.sleep(1)
                self.motion_home_test()

                while not (A_ZONE or B_ZONE or C_ZONE):  # 캡슐 인식 대기
                    time.sleep(0.2)
                    print('캡슐 인식 대기중...')
                time.sleep(2)

                self.motion_grab_capsule_test()
                self.motion_check_sealing_test()

                count = 0
                while True:
                    # if sealing_check request arrives or 5sec past
                    if self.NOT_SEAL or count >= 5:      # 5초 간 씰 인식
                        print('seal check complete')
                        break
                    time.sleep(0.2)
                    count += 0.2

                if self.NOT_SEAL:
                    self.motion_place_capsule_test()
                    self.motion_grab_cup_test()
                    self.motion_topping_test(order)
                    self.motion_make_icecream_test()
                    self.motion_serve_test()
                    self.motion_trash_capsule_test()
                    self.motion_home_test()
                    print('icecream finish')
                else:
                    self.motion_place_fail_capsule_test()
                    self.motion_home_test()
                    self.order_list.insert(0, raw_order)
                    print('please take off the seal')

                code = self._arm.stop_lite6_gripper()
                if not self._check_code(code, 'stop_lite6_gripper'):
                    return
                self.A_ZONE, self.B_ZONE, self.C_ZONE, self.NOT_SEAL = False, False, False, False
                self.A_ZONE_start_time, self.B_ZONE_start_time, self.C_ZONE_start_time = None, None, None
                self.start_time = None
                time.sleep(3)   
            elif self.MODE == 'gritting':
                self.gritting(gender)           


if __name__ == '__main__':
    RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
    arm = XArmAPI('192.168.1.167', baud_checkset=False)
    robot_main = RobotMain(arm)
    yolo_main = YOLOMain(robot_main)

    # 스레드 생성
    robot_thread = threading.Thread(target=robot_main.run_robot)
    yolo_thread = threading.Thread(target=yolo_main.segmentation)

    # 스레드 시작
    robot_thread.start()
    yolo_thread.start()

    # 스레드가 끝날 때까지 대기
    robot_thread.join()
    yolo_thread.join()