from os.path import join as pjoin
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import textwrap



def plot_3d_motion(save_path, kinematic_tree, joints, title, figsize=(10, 10), fps=120, radius=4):
    """
    3D 모션 데이터를 애니메이션으로 저장하는 함수 (수정 및 완성 버전)
    """
    
    # 1. 데이터 전처리
    data = joints.copy().reshape(len(joints), -1, 3)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    # 지면과 평행하게 만들기 위해 높이 오프셋 제거
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    
    # 루트 조인트를 기준으로 한 궤적(trajectory) 계산
    trajec = data[:, 0, [0, 2]].copy() # 궤적 데이터는 복사해서 사용

    # 루트 조인트를 원점으로 이동시켜 모션 자체에 집중
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    # 2. 플롯 및 애니메이션 설정
    fig = plt.figure(figsize=figsize)
    # 최신 Matplotlib API 권장 사항
    ax = fig.add_subplot(111, projection='3d')
    
    # 긴 제목 처리
    wrapped_title = textwrap.fill(title, width=40)
    fig.suptitle(wrapped_title, fontsize=16)

    # 축 범위와 시점 설정
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([0, radius])
    ax.set_zlim3d([0, radius])
    ax.view_init(elev=120, azim=-90)
    ax.dist = 7.5
    ax.grid(b=False)
    plt.axis('off')

    # 애니메이션 객체들을 담을 리스트
    lines = []
    # 궤적을 그릴 선 객체 (초기화 시 빈 데이터로 생성)
    traj_line, = ax.plot([], [], [], 'b-', linewidth=1.0) 
    
    # 스켈레톤의 각 체인에 대한 선 객체 생성
    # 색상 리스트가 kinematic_tree보다 짧아도 에러가 나지 않도록 itertools.cycle 사용 가능
    # 여기서는 간단하게 나머지 연산자로 처리
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    
    for i, chain in enumerate(kinematic_tree):
        color = colors[i % len(colors)]
        linewidth = 4.0 if i < 5 else 2.0
        # 비어있는 선(line) 객체를 미리 생성해둠
        line, = ax.plot([], [], [], linewidth=linewidth, color=color)
        lines.append(line)

    # 지면(plane)을 그리는 함수
    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [[minx, miny, minz], [minx, miny, maxz], [maxx, miny, maxz], [maxx, miny, minz]]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        return xz_plane

    # 초기 지면 생성 (애니메이션 루프에서 지워지고 다시 그려짐)
    plane_collection = plot_xzPlane(MINS[0], MAXS[0], 0, MINS[2], MAXS[2])
    ax.add_collection3d(plane_collection)
    
    frame_number = data.shape[0]

    def update(index):
        nonlocal plane_collection # 외부 함수의 객체를 수정하기 위해 nonlocal 선언
        
        # 1. 스켈레톤 업데이트 (가장 중요한 부분)
        # 미리 생성된 line 객체들의 데이터만 교체 (set_data, set_3d_properties)
        for i, (line, chain) in enumerate(zip(lines, kinematic_tree)):
            x = data[index, chain, 0]
            y = data[index, chain, 1]
            z = data[index, chain, 2]
            line.set_data(x, y)
            line.set_3d_properties(z)

        # 2. 궤적 업데이트
        # 현재 프레임을 기준으로 이전 궤적을 그림
        current_traj_x = trajec[:index+1, 0] - trajec[index, 0]
        current_traj_z = trajec[:index+1, 1] - trajec[index, 1]
        traj_line.set_data(current_traj_x, np.zeros_like(current_traj_x))
        traj_line.set_3d_properties(current_traj_z)

        # 3. 지면 위치 업데이트 (지우고 다시 그리는 방식이 간단함)
        plane_collection.remove() # 이전 프레임의 지면 제거
        plane_collection = plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, 
                                        MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        ax.add_collection3d(plane_collection)

        return lines + [traj_line, plane_collection]

    # FuncAnimation 생성
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, blit=False)

    # 파일 저장 (writer 명시 권장)
    try:
        # mp4 저장 시 ffmpeg writer를 사용하는 것이 안정적입니다.
        ani.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("FFMpeg가 설치되어 있는지, 환경 변수(PATH)에 등록되어 있는지 확인하세요.")
        print("설치 방법: 'conda install ffmpeg' 또는 'sudo apt-get install ffmpeg'")
    
    plt.close(fig) # 메모리 누수 방지를 위해 figure를 명시적으로 닫아줍니다.

