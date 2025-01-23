#pragma once
#define PCL_NO_PRECOMPILE

#include <iostream>
#include <fstream>

#include <ros/ros.h>

#include <Eigen/Dense>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>

#include <pcl/kdtree/kdtree_flann.h>

#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

// 根据是否定义USE_EVALUATION_POINT_TYPE宏来设置是否使用评估点类型
constexpr bool isEvaluationPointType =
#ifdef USE_EVALUATION_POINT_TYPE
    true;
#else
    false;
#endif

#ifndef _POINT_TYPE_DEFINITION_
#define _POINT_TYPE_DEFINITION_
// 定义带有标签的点云类型,用于评估
struct EIGEN_ALIGN16 PointXYZIRTRGBL
{
    PCL_ADD_POINT4D;               // 添加XYZ坐标
    PCL_ADD_INTENSITY;             // 添加强度信息
    std::uint16_t ring;            // 激光雷达线束编号
    float time;                    // 时间戳
    PCL_ADD_RGB;                   // 添加RGB颜色信息
    std::uint16_t label;           // 点的标签信息
    PCL_MAKE_ALIGNED_OPERATOR_NEW; // 确保内存对齐
};

// 注册点云结构体,定义各字段
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRTRGBL,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint16_t, ring, ring)(float, time, time)(std::uint32_t, rgb, rgb)(std::uint16_t, label, label))

#ifdef USE_EVALUATION_POINT_TYPE
// 评估模式下使用带标签的点云类型
using PointType = PointXYZIRTRGBL;
using PointTypeMOS = PointXYZIRTRGBL;
using PointTypeMOSEval = PointXYZIRTRGBL;
using PointTypeMOSDeskew = PointXYZIRTRGBL;
#else
// 定义普通点云类型,用于正常使用
struct PointXYZIRGB
{
    PCL_ADD_POINT4D;                 // XYZ坐标
    PCL_ADD_INTENSITY;               // 强度信息
    PCL_ADD_RGB;                     // RGB颜色信息
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 内存对齐
} EIGEN_ALIGN16;

// 注册普通点云结构体
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRGB,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(std::uint32_t, rgb, rgb))

// 正常模式下使用不同的点云类型
using PointType = pcl::PointXYZI;           // 基本点云类型
using PointTypeMOS = PointXYZIRGB;          // MOS处理用点云类型
using PointTypeMOSEval = PointXYZIRTRGBL;   // MOS评估用点云类型
using PointTypeMOSDeskew = PointXYZIRTRGBL; // 去畸变用点云类型
#endif
#endif

// ScanFrame结构体用于存储一帧激光雷达扫描的相关数据
struct ScanFrame
{
    pcl::PointCloud<PointTypeMOS>::Ptr m_scan;         // 点云数据指针
    pcl::PointCloud<PointTypeMOS *>::Ptr m_scan_ptrs;  // 点云数据指针的指针
    Eigen::Affine3f m_tf_frame_to_map;                 // 从当前帧到地图坐标系的变换矩阵
    std::shared_ptr<std::vector<float>> m_range_image; // 距离图像数据
    int m_frame_id;                                    // 帧ID
    int m_keyframe_id;                                 // 关键帧ID
    double m_time_s;                                   // 时间戳(秒)

    // 默认构造函数,初始化所有成员为空或默认值
    ScanFrame() : m_scan(nullptr),
                  m_scan_ptrs(nullptr),
                  m_tf_frame_to_map(Eigen::Affine3f()),
                  m_range_image(nullptr),
                  m_frame_id(-1),
                  m_keyframe_id(-1),
                  m_time_s(-1) {}

    // 带参数的构造函数,用于初始化所有成员变量
    ScanFrame(pcl::PointCloud<PointTypeMOS>::Ptr i_scan,
              pcl::PointCloud<PointTypeMOS *>::Ptr i_scan_ptrs,
              Eigen::Affine3f &i_tf_frame_to_map,
              std::shared_ptr<std::vector<float>> i_range_image,
              int &i_frame_id,
              int &i_keyframe_id,
              double &i_time_s) : m_scan(i_scan),
                                  m_scan_ptrs(i_scan_ptrs),
                                  m_tf_frame_to_map(i_tf_frame_to_map),
                                  m_range_image(i_range_image),
                                  m_frame_id(i_frame_id),
                                  m_keyframe_id(i_keyframe_id),
                                  m_time_s(i_time_s) {}

    // 拷贝构造函数,用于对象复制
    ScanFrame(const ScanFrame &other)
    {
        m_scan = other.m_scan;
        m_scan_ptrs = other.m_scan_ptrs;
        m_tf_frame_to_map = other.m_tf_frame_to_map;
        m_range_image = other.m_range_image;
        m_frame_id = other.m_frame_id;
        m_keyframe_id = other.m_keyframe_id;
        m_time_s = other.m_time_s;
    }

    // 赋值运算符重载,用于对象赋值
    ScanFrame &operator=(const ScanFrame &other)
    {
        if (this == &other)
            return *this;
        m_scan = other.m_scan;
        m_scan_ptrs = other.m_scan_ptrs;
        m_tf_frame_to_map = other.m_tf_frame_to_map;
        m_range_image = other.m_range_image;
        m_frame_id = other.m_frame_id;
        m_keyframe_id = other.m_keyframe_id;
        m_time_s = other.m_time_s;
        return *this;
    }

    // 析构函数
    ~ScanFrame() {}
};

// AWV_MOS类定义了一个用于移动物体分割(Moving Object Segmentation)的系统
class AWV_MOS
{
public:
    // 构造和析构函数
    AWV_MOS();
    ~AWV_MOS();

public:
    // 在线MOS处理函数,用于实时处理单帧点云数据
    void RunOnlineMOS(const pcl::PointCloud<PointTypeMOS>::Ptr &i_scan,
                      const Eigen::Affine3f &i_tf_frame_to_map,
                      const int &i_frame_id,
                      const double &i_time_s,
                      const bool &i_is_keyframe = false,
                      const bool &i_is_prior = false);

    // 静态地图构建函数,处理一系列点云数据生成静态和动态地图
    void RunStaticMapping(const std::vector<pcl::PointCloud<PointTypeMOS>::Ptr> &i_scans,
                          const std::vector<Eigen::Affine3f> &i_poses,
                          const std::vector<int> &i_frames_id,
                          const std::vector<double> &i_times,
                          pcl::PointCloud<PointTypeMOS>::Ptr &o_static_map,
                          pcl::PointCloud<PointTypeMOS>::Ptr &o_dynamic_map);

public:
    // 工具函数
    void SaveConfigParams();                                                                          // 保存配置参数
    bool KeyframeSelection(const Eigen::Affine3f &i_tf_frame_to_map, const double &i_time_scanframe); // 关键帧选择
    bool KeyframeSelection(const Eigen::Affine3f &i_tf_prev_frame_to_map, const double &i_time_prev_scanframe, const Eigen::Affine3f &i_tf_pres_frame_to_map, const double &i_time_pres_scanframe);
    void Reset();                                                                                                     // 重置系统
    void WritePrediction(const pcl::PointCloud<PointTypeMOS>::Ptr &i_segmented_scan, const std::string &i_save_path); // 保存预测结果
    void GetSegmentedScan(pcl::PointCloud<PointTypeMOS>::Ptr &o_segmented_scan);                                      // 获取分割后的点云

public:
    ros::NodeHandle nh; // ROS节点句柄

    // AWV_MOS配置参数
    // - Topic命名空间
    std::string m_cfg_s_output_pc_namespace;
    // - 激光雷达特性参数
    float m_cfg_f_lidar_horizontal_resolution_deg;    // 水平分辨率
    float m_cfg_f_lidar_vertical_resolution_deg;      // 垂直分辨率
    float m_cfg_f_lidar_vertical_fov_upper_bound_deg; // 垂直视场角上限
    float m_cfg_f_lidar_vertical_fov_lower_bound_deg; // 垂直视场角下限
    // - 先验MOS更新选项
    bool m_cfg_b_use_prior_mos;             // 是否使用先验MOS
    float m_cfg_f_imu_odom_trans_err_std_m; // IMU里程计平移误差标准差
    float m_cfg_f_imu_odom_rot_err_std_rad; // IMU里程计旋转误差标准差
    // - 参考帧参数
    float m_cfg_f_keyframe_translation_threshold_m; // 关键帧平移阈值
    float m_cfg_f_keyframe_rotation_threshold_rad;  // 关键帧旋转阈值
    float m_cfg_f_keyframe_time_threshold_s;        // 关键帧时间阈值
    bool m_cfg_b_use_ref_frame_instant_charge;      // 是否使用参考帧即时更新
    int m_cfg_n_mos_ref_frame_size;                 // MOS参考帧大小
    // - 点到窗口比较参数
    float m_cfg_f_meas_range_std_m;                               // 测量距离标准差
    float m_cfg_f_meas_theta_std_rad;                             // 测量角度标准差
    float m_cfg_f_meas_phi_std_rad;                               // 测量方位角标准差
    float m_cfg_f_scan_matching_trans_err_std_m;                  // 扫描匹配平移误差标准差
    float m_cfg_f_scan_matching_rot_err_std_rad;                  // 扫描匹配旋转误差标准差
    float m_cfg_f_range_image_observation_window_u_angle_deg_min; // 观测窗口最小U角度
    float m_cfg_f_range_image_observation_window_u_angle_deg_max; // 观测窗口最大U角度
    float m_cfg_f_range_image_observation_window_v_angle_deg_min; // 观测窗口最小V角度
    float m_cfg_f_range_image_observation_window_v_angle_deg_max; // 观测窗口最大V角度
    float m_cfg_f_range_image_z_correction;                       // 深度图Z轴校正
    float m_cfg_f_range_image_min_dist_m;                         // 深度图最小距离
    float m_cfg_f_range_image_min_height_m;                       // 深度图最小高度
    bool m_cfg_b_use_range_image_noise_filtering;                 // 是否使用深度图噪声滤波
    float m_cfg_f_range_image_noise_filtering_min_diff_m;         // 深度图噪声滤波最小差值
    // - 运动置信度计算参数
    float m_cfg_f_moving_confidence; // 运动置信度
    float m_cfg_f_static_confidence; // 静态置信度
    // - 物体尺度测试参数
    bool m_cfg_b_use_object_scale_test;                    // 是否使用物体尺度测试
    float m_cfg_f_object_scale_test_valid_visible_range_m; // 有效可见范围
    float m_cfg_f_object_scale_test_min_height_m;          // 最小高度
    float m_cfg_f_object_scale_test_min_visible_area_m2;   // 最小可见area
    float m_cfg_f_object_scale_test_point_search_radius_m; // 点搜索半径
    // - 区域生长参数
    bool m_cfg_b_use_region_growing;                     // 是否使用区域生长
    float m_cfg_f_region_growing_voxel_leaf_size_m;      // 体素大小
    float m_cfg_f_region_growing_max_iteration;          // 最大迭代次数
    float m_cfg_f_region_growing_point_search_radius_m;  // 点搜索半径
    float m_cfg_f_region_growing_ground_filter_height_m; // 地面滤波高度
    // - 扫描匹配权重参数
    float m_cfg_f_static_weight_ratio; // 静态权重比例
    // - ROS消息发布参数
    bool m_cfg_b_publish_pc; // 是否发布点云
    // - CPU参数
    int m_cfg_n_num_cpu_cores; // CPU核心数
    // - 预测结果写入
    bool m_cfg_b_use_prediction_write; // 是否写入预测结果
    // - 建图参数
    int m_cfg_i_mapping_start_frame_limit;           // 起始帧限制
    int m_cfg_i_mapping_end_frame_limit;             // 结束帧限制
    bool m_cfg_b_mapping_use_save_map;               // 是否保存地图
    bool m_cfg_b_mapping_use_only_keyframes;         // 是否只使用关键帧
    bool m_cfg_b_mapping_use_mos_backward_update;    // 是否使用MOS反向更新
    float m_cfg_f_mapping_section_division_length_m; // 地图分段长度
    float m_cfg_f_mapping_voxel_leaf_size_m;         // 体素大小
    bool m_cfg_b_mapping_use_visualization;          // 是否使用可视化
    // - 日志写入参数
    std::string m_log_write_folder_path; // 日志文件夹路径
    // 配置文件路径
    std::string m_config_file_path;     // 配置文件路径
    std::string m_mos_config_file_path; // MOS配置文件路径

private:
    // 常量定义
    static constexpr float RADtoDEG = 180. / M_PI; // 弧度转角度
    static constexpr float DEGtoRAD = M_PI / 180.; // 角度转弧度

    // 数据缓存
    std::deque<ScanFrame> m_deq_reference_frame_buffer; // 参考帧缓存
    ScanFrame m_query_frame;                            // 查询帧

    // 点云处理工具
    pcl::KdTreeFLANN<PointTypeMOS>::Ptr m_kdtree_scan_moving;        // 运动点云KD树
    pcl::KdTreeFLANN<PointTypeMOS>::Ptr m_kdtree_scan_unknown;       // 未知点云KD树
    pcl::VoxelGrid<PointTypeMOS> m_voxel_grid_filter_region_growing; // 区域生长体素滤波器

    // 深度图参数
    int m_num_range_image_cols;   // 列数
    int m_num_range_image_rows;   // 行数
    int m_num_range_image_pixels; // 像素总数

private:
    // 帧管理
    std::shared_ptr<std::deque<ScanFrame>> m_deq_frames_container; // 帧容器
    std::vector<int> m_vec_keyframe_frame_id_list;                 // 关键帧ID列表
    std::vector<int> m_vec_keyframe_frame_index_list;              // 关键帧索引列表

private:
    // 核心处理函数
    ScanFrame SegmentMovingObject(const pcl::PointCloud<PointTypeMOS>::Ptr &i_scan, const Eigen::Affine3f &i_tf_frame_to_map, const int &i_frame_id, const double &i_time_s, const bool &i_is_keyframe, const bool &i_is_prior);
    void ObjectScaleTest(ScanFrame &i_frame);                               // 物体尺度测试
    void RegionGrowing(ScanFrame &i_frame);                                 // 区域生长
    void ManageBuffer(const ScanFrame &i_frame, const bool &i_is_keyframe); // 缓存管理

    // 静态地图构建相关函数
    ScanFrame InitScanFrame(const pcl::PointCloud<PointTypeMOS>::Ptr &i_scan, const Eigen::Affine3f &i_tf_frame_to_map, const int &i_frame_id, const double &i_time_s, const bool &i_is_keyframe);
    void SelectReferenceFrames(const ScanFrame &i_query_frame, std::deque<ScanFrame> &o_reference_frames);
    void SegmentMovingObject(const ScanFrame &i_query_frame, const std::deque<ScanFrame> &i_reference_frames);
    void VoxelDownSamplingPreservingLabels(const pcl::PointCloud<PointTypeMOSEval>::Ptr &i_src, const float &i_voxel_leaf_size_m, pcl::PointCloud<PointTypeMOSEval> &o_dst);
    void VoxelDownSamplingPreservingLabelsLargeScale(const pcl::PointCloud<PointTypeMOSEval>::Ptr &i_src, const float &i_section_division_length_m, const float &i_voxel_leaf_size_m, pcl::PointCloud<PointTypeMOSEval> &o_dst);
    void VoxelDownSamplingLargeScale(const pcl::PointCloud<PointTypeMOS>::Ptr &i_src, const float &i_section_division_length_m, const float &i_voxel_leaf_size_m, pcl::PointCloud<PointTypeMOS> &o_dst);
    void EvaluteMap(pcl::PointCloud<PointTypeMOSEval>::Ptr &i_src);

private:
    // 辅助处理函数
    void RangeImageNoiseFiltering(const std::shared_ptr<std::vector<float>> &i_range_image, std::shared_ptr<std::vector<float>> &o_range_image_filtered);
    float PointToWindowComparision(const std::shared_ptr<std::vector<float>> &i_range_image,
                                   const Eigen::Vector3f &i_spherical_point_m_rad,
                                   const int &i_winow_size_u,
                                   const int &i_winow_size_v,
                                   const float &i_observavle_radial_distance_range);
    std::array<unsigned char, 3> DempsterCombination(const std::array<unsigned char, 3> i_src_belief, std::array<unsigned char, 3> const i_other_belief);
};