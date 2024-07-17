import os
import shutil
import imagehash
from PIL import Image
from tqdm import tqdm

def is_image_file(file_path):
    """파일이 유효한 이미지 파일인지 확인."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    return os.path.splitext(file_path)[1].lower() in valid_extensions

def calculate_image_hash(image_path):
    """이미지의 지각적 해시를 계산."""
    try:
        image = Image.open(image_path)
        return imagehash.average_hash(image)
    except Exception as e:
        print(f"Error calculating hash for {image_path}: {e}")
        return None

def remove_non_image_files(directory):
    """디렉토리에서 이미지 파일이 아닌 파일을 제거."""
    removed_files = []
    for root, _, files in os.walk(directory):  # 디렉토리 내 모든 파일을 탐색
        for file in files:  # 각 파일에 대해
            file_path = os.path.join(root, file)  # 파일의 전체 경로 생성
            if not is_image_file(file_path):  # 파일이 이미지 파일이 아니면
                try:
                    os.remove(file_path)  # 파일을 삭제
                    removed_files.append(file_path)  # 삭제된 파일 경로를 리스트에 추가
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
    return removed_files

def find_duplicates_and_move(directory, dest_directory):
    """중복된 이미지를 찾아서 대상 디렉토리로 이동."""
    hashes = {}  # 해시 값을 저장할 딕셔너리
    duplicates = []  # 중복된 파일 경로를 저장할 리스트

    # 처리할 파일의 총 수를 계산
    total_files = sum(len(files) for _, _, files in os.walk(directory))

    # 진행 상황을 추적하기 위해 tqdm 사용
    with tqdm(total=total_files, unit="file") as pbar:
        for root, _, files in os.walk(directory):  # 디렉토리 내 모든 파일을 탐색
            for file in files:  # 각 파일에 대해
                file_path = os.path.join(root, file)  # 파일의 전체 경로 생성
                if is_image_file(file_path):  # 파일이 이미지 파일이면
                    try:
                        file_hash = calculate_image_hash(file_path)  # 파일의 해시 값을 계산
                        if file_hash:
                            if file_hash in hashes:  # 해시 값이 이미 존재하면 중복 파일로 간주
                                duplicates.append(file_path)
                            else:  # 새로운 해시 값이면 딕셔너리에 추가
                                hashes[file_hash] = file_path
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                pbar.update(1)  # 진행 상황 업데이트

    # 중복 파일을 대상 디렉토리로 이동
    if not os.path.exists(dest_directory):  # 대상 디렉토리가 없으면 생성
        os.makedirs(dest_directory)
    
    moved_files = []
    for file in duplicates:  # 중복 파일들을
        try:
            shutil.move(file, dest_directory)  # 이동
            moved_files.append(file)  # 이동된 파일 경로를 리스트에 추가
        except Exception as e:
            print(f"Error moving file {file} to {dest_directory}: {e}")
    return moved_files

def sort_images_by_similarity(directory):
    """디렉토리 내의 이미지를 지각적 해시 유사성에 따라 정렬."""
    images = []
    
    for root, _, files in os.walk(directory):  # 디렉토리 내 모든 파일을 탐색
        for file in files:  # 각 파일에 대해
            file_path = os.path.join(root, file)  # 파일의 전체 경로 생성
            if is_image_file(file_path):  # 파일이 이미지 파일이면
                try:
                    file_hash = calculate_image_hash(file_path)  # 파일의 해시 값을 계산
                    if file_hash:
                        images.append((file_path, file_hash))  # 이미지 경로와 해시 값을 리스트에 추가
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    # 해시 값을 문자열로 변환하여 이미지 정렬
    images.sort(key=lambda x: str(x[1]))

    renamed_files = []
    for index, (file_path, _) in enumerate(images):  # 정렬된 이미지를 순서대로
        file_extension = os.path.splitext(file_path)[1]  # 파일 확장자를 추출
        new_name = f"{index:04d}{file_extension}"  # 새로운 이름을 생성
        new_file_path = os.path.join(directory, new_name)  # 새로운 파일 경로 생성
        try:
            os.rename(file_path, new_file_path)  # 파일 이름 변경
            renamed_files.append((file_path, new_file_path))  # 변경된 파일 경로를 리스트에 추가
        except Exception as e:
            print(f"Error renaming file {file_path} to {new_file_path}: {e}")
    return renamed_files

def main():
    # 경로를 여기서 명확히 지정
    source_directory = "/home/hui/Downloads/annoyed1"  # 원본 이미지 디렉토리 경로
    duplicates_directory = "/home/hui/Downloads/annoyed2"  # 중복 이미지가 이동될 디렉토리 경로

    # Step 1: Remove non-image files
    removed_files = remove_non_image_files(source_directory)  # 이미지 파일이 아닌 파일을 제거
    print(f"Step 1 completed: Removed {len(removed_files)} non-image files.")  # 제거된 파일 수 출력

    # Step 2: Find and move duplicate images
    moved_files = find_duplicates_and_move(source_directory, duplicates_directory)  # 중복된 이미지를 찾아 이동
    print(f"Step 2 completed: Moved {len(moved_files)} duplicate files to {duplicates_directory}.")  # 이동된 파일 수 출력

    # Step 3: Sort remaining images by similarity
    renamed_files = sort_images_by_similarity(source_directory)  # 남은 이미지를 유사성에 따라 정렬
    print(f"Step 3 completed: Renamed {len(renamed_files)} files for similarity sorting.")  # 정렬된 파일 수 출력

    # Detailed logs for verification
    if removed_files:  # 제거된 파일이 있으면
        print("Removed non-image files:")  # 제거된 파일 목록 출력
        for file in removed_files:
            print(f"  - {file}")

    if moved_files:  # 이동된 파일이 있으면
        print("Moved duplicate files:")  # 이동된 파일 목록 출력
        for file in moved_files:
            print(f"  - {file}")

    if renamed_files:  # 이름이 변경된 파일이 있으면
        print("Renamed files:")  # 이름이 변경된 파일 목록 출력
        for old_path, new_path in renamed_files:
            print(f"  - {old_path} to {new_path}")

# Example usage
if __name__ == "__main__":
    main()

