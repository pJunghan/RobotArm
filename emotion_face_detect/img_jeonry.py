import os
import uuid
import shutil

def rename_and_move_images_in_folder(folder_path, folder_number, dest_directory):
    """특정 폴더 안의 모든 이미지를 고유 이름으로 변경하고 이동."""
    for root, _, files in os.walk(folder_path):  # 주어진 폴더 경로의 모든 파일을 탐색
        for file in files:  # 각 파일에 대해
            file_path = os.path.join(root, file)  # 파일의 전체 경로 생성
            file_extension = os.path.splitext(file)[1]  # 파일 확장자 추출
            new_name = f"{folder_number}_{uuid.uuid4()}{file_extension}"  # 폴더 번호와 고유 ID를 조합하여 새 이름 생성
            new_file_path = os.path.join(dest_directory, new_name)  # 새 파일 경로 생성
            os.rename(file_path, new_file_path)  # 파일을 새 이름으로 이동

def rename_and_move_images_in_all_folders(base_directory, dest_directory):
    """기본 디렉토리의 모든 하위 폴더에서 이미지를 고유 이름으로 변경하고 대상 디렉토리로 이동."""
    if not os.path.exists(dest_directory):  # 대상 디렉토리가 없으면 생성
        os.makedirs(dest_directory)

    for i in range(1, 18):  # 1부터 17까지의 숫자로 폴더 이름을 생성
        folder_path = os.path.join(base_directory, str(i))  # 각 숫자에 해당하는 폴더 경로 생성
        if os.path.exists(folder_path):  # 폴더가 존재하면
            rename_and_move_images_in_folder(folder_path, i, dest_directory)  # 이미지를 이름 변경 후 이동
        else:  # 폴더가 존재하지 않으면
            print(f"Folder {folder_path} does not exist.")  # 폴더가 없다는 메시지 출력

# 예제 사용법
if __name__ == "__main__":
    base_directory = "/home/hui/Downloads/ImageAssistant_Batch_Image_Downloader"  # 원본 이미지 폴더 경로
    destination_directory = "/home/hui/Downloads/annoyed1"  # 이미지가 이동될 대상 경로
    rename_and_move_images_in_all_folders(base_directory, destination_directory)  # 함수 실행

