﻿
#include "ggl.h"
#include "MPASOReader.h"
#include "MPASOGrid.h"
#include "MPASOSolution.h"
#include "MPASOField.h"
#include "ImageBuffer.hpp"
#include "MPASOVisualizer.h"
#include "VTKFileManager.hpp"
#include "cxxopts.hpp"
#include "ndarray/ndarray_group_stream.hh"

// time fixed
std::shared_ptr<MPASOGrid> mpasoGrid = nullptr;
std::shared_ptr<MPASOSolution> mpasoSol = nullptr;
std::shared_ptr<MPASOField> mpasoField = nullptr;

// time varying
std::shared_ptr<MPASOGrid> tv_mpasoGrid = nullptr;
std::shared_ptr<MPASOSolution> tv_mpasoSol1 = nullptr;
std::shared_ptr<MPASOSolution> tv_mpasoSol2 = nullptr;
std::shared_ptr<MPASOField> tv_mpasoField1 = nullptr;
std::shared_ptr<MPASOField> tv_mpasoField2 = nullptr;

const char* path;

std::string input_yaml_filename;
std::string data_path_prefix;

std::string vectorToString(const std::vector<std::string>& vec, const std::string& delimiter = ", ") {
	std::ostringstream oss;
	for (size_t i = 0; i < vec.size(); ++i) {
		if (i != 0) {
			oss << delimiter;
		}
		oss << vec[i];
	}
	return oss.str();
}


void TimeFixed(sycl::queue& sycl_Q, ftk::stream* stream)
{
	auto grid_path = stream->substreams[0]->filenames[0];
	auto vel_path = stream->substreams[1]->filenames[0];
	
	auto gs = stream->read_static();
	std::cout << " [ files attribute information" << " ]\n";
	mpasoGrid = std::make_shared<MPASOGrid>();
	mpasoGrid->initGrid(gs.get(), MPASOReader::readGridInfo(grid_path).get());
	mpasoGrid->createKDTree("./index.bin", sycl_Q);

	int timestep = 0;
	auto gt = stream->read(timestep);
	mpasoSol = std::make_shared<MPASOSolution>();
	mpasoSol->initSolution(gt.get(), MPASOReader::readSolInfo(vel_path, timestep).get());
	mpasoSol->calcCellVertexZtop(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellCenterVelocity(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), sycl_Q);

	mpasoField = std::make_shared<MPASOField>();
	mpasoField->initField(mpasoGrid, mpasoSol);


	constexpr int width = 361; constexpr int height = 181;
	VisualizationSettings* config = new VisualizationSettings();
	config->imageSize = vec2(width, height);
	config->LatRange = vec2(20.0, 50.0);
	config->LonRange = vec2(-17.0, 17.0);
	config->FixedLayer = 0.0;
	config->TimeStep = 0.0;
	config->CalcType = CalcAttributeType::kZonalMerimoal;
	config->VisType = VisualizeType::kFixedLayer;
	config->PositionType = CalcPositionType::kPoint;


	VisualizationSettings* config_fixed_depth = new VisualizationSettings();
	config_fixed_depth->imageSize = vec2(width, height);
	config_fixed_depth->LatRange = vec2(20.0, 50.0);
	config_fixed_depth->LonRange = vec2(-17.0, 17.0);
	config_fixed_depth->FixedDepth = 700.0;
	config_fixed_depth->TimeStep = 0.0;
	config_fixed_depth->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_depth->VisType = VisualizeType::kFixedDepth;
	config_fixed_depth->PositionType = CalcPositionType::kPoint;

	VisualizationSettings* config_fixed_lat = new VisualizationSettings();
	config_fixed_lat->imageSize = vec2(width, height);
	config_fixed_lat->LonRange = vec2(-17.0, 17.0);
	config_fixed_lat->DepthRange = vec2(0.0, 5000.0);
	config_fixed_lat->FixedLatitude = 35.0;
	config_fixed_lat->TimeStep = 0.0;
	config_fixed_lat->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_lat->VisType = VisualizeType::kFixedDepth;
	config_fixed_lat->PositionType = CalcPositionType::kPoint;

	std::cout << "==========================================\n";

	ImageBuffer<double>* img1 = new ImageBuffer<double>(config->imageSize.x(), config->imageSize.y());
	ImageBuffer<double>* img2 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
	ImageBuffer<double>* img3 = new ImageBuffer<double>(config_fixed_lat->imageSize.x(), config_fixed_lat->imageSize.y());

	// remapping
	std::cout << " [ Run remapping test(fixed time)" << " ]\n";
	std::cout << " 	[ Run remapping VisualizeFixedLayer(fixed time)" << " ]\n";
	MPASOVisualizer::VisualizeFixedLayer(mpasoField.get(), config, img1, sycl_Q);
	VTKFileManager::SaveVTI(img1, config);
	
	std::cout << " 	[ Run remapping VisualizeFixedDepth(fixed time)" << " ]\n";
	MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config_fixed_depth, img2, sycl_Q);
	VTKFileManager::SaveVTI(img2, config_fixed_depth);
	//MPASOVisualizer::VisualizeFixedLatitude(mpasoField.get(), config_fixed_lat, img3, sycl_Q);
  	//Debug("finished... fixed_lat_35.000000.vti");

	std::cout << " 	[ Run Trtrajector(fixed time)" << " ]\n";
	SamplingSettings* sample_conf = new SamplingSettings;
	sample_conf->sampleDepth = config_fixed_depth->FixedDepth;
	sample_conf->sampleLatitudeRange = vec2(25.0, 45.0);
	sample_conf->sampleLongitudeRange = vec2(-10.0, 10.0);
	sample_conf->sampleNumer = vec2i(31, 31);

	TrajectorySettings* traj_conf = new TrajectorySettings;
	traj_conf->deltaT = ONE_MINUTE;			
	traj_conf->simulationDuration = ONE_MINUTE * 30;	
	traj_conf->recordT = ONE_MINUTE * 6;
	traj_conf->fileName = "output_line";

	std::vector<CartesianCoord> sample_points; std::vector<int> cell_id_vec;
	MPASOVisualizer::GenerateSamplePoint(sample_points, sample_conf);
	//std::cout << "points size() " << sample_points.size() << std::endl;
	VTKFileManager::SavePointAsVTP(sample_points, "output_origin");
	
	mpasoField->calcInWhichCells(sample_points, cell_id_vec);
	//std::cout << std::fixed << std::setprecision(4) << "before trajector cell_id [ " << cell_id_vec[0] << " ]" << " " << sample_points[0].x() << " " << sample_points[0].y() << " " << sample_points[0].z() << std::endl;

	MPASOVisualizer::VisualizeTrajectory(mpasoField.get(), sample_points, traj_conf, cell_id_vec, sycl_Q);

}


void TimeVarying(sycl::queue& sycl_Q, 
	std::vector<std::string>& grid_path_vec, std::vector<std::string>& vel_path_vec)
{
	tv_mpasoGrid = std::make_shared<MPASOGrid>();
	tv_mpasoGrid->initGrid(MPASOReader::loadingGridInfo(grid_path_vec[0]).get());
	tv_mpasoGrid->createKDTree("./index1.bin", sycl_Q);

	tv_mpasoSol1 = std::make_shared<MPASOSolution>();
	tv_mpasoSol1->initSolution(MPASOReader::loadingVelocityInfo(vel_path_vec[0], 0).get());
	//mpasoSol->calcCellCenterZtop();
	tv_mpasoSol1->calcCellVertexZtop(tv_mpasoGrid.get(), sycl_Q);
	tv_mpasoSol1->calcCellCenterVelocity(tv_mpasoGrid.get(), sycl_Q);
	tv_mpasoSol1->calcCellVertexVelocity(tv_mpasoGrid.get(), sycl_Q);

	tv_mpasoSol2 = std::make_shared<MPASOSolution>();
	tv_mpasoSol2->initSolution(MPASOReader::loadingVelocityInfo(vel_path_vec[4], 0).get());
	//mpasoSol->calcCellCenterZtop();
	tv_mpasoSol2->calcCellVertexZtop(tv_mpasoGrid.get(), sycl_Q);
	tv_mpasoSol2->calcCellCenterVelocity(tv_mpasoGrid.get(), sycl_Q);
	tv_mpasoSol2->calcCellVertexVelocity(tv_mpasoGrid.get(), sycl_Q);

	tv_mpasoField1 = std::make_shared<MPASOField>();
	tv_mpasoField1->initField(tv_mpasoGrid, tv_mpasoSol1);

	tv_mpasoField2 = std::make_shared<MPASOField>();
	tv_mpasoField2->initField(tv_mpasoGrid, tv_mpasoSol2);

	constexpr int width = 361; constexpr int height = 181;
	VisualizationSettings* config = new VisualizationSettings();
	config->imageSize = vec2(width, height);
	config->LatRange = vec2(20.0, 50.0);
	config->LonRange = vec2(-17.0, 17.0);
	config->FixedLayer = 0.0;
	config->TimeStep = 0.0;
	config->CalcType = CalcAttributeType::kZonalMerimoal;
	config->VisType = VisualizeType::kFixedLayer;
	config->PositionType = CalcPositionType::kPoint;

	VisualizationSettings* config_fixed_depth = new VisualizationSettings();
	config_fixed_depth->imageSize = vec2(width, height);
	config_fixed_depth->LatRange = vec2(20.0, 50.0);
	config_fixed_depth->LonRange = vec2(-17.0, 17.0);
	config_fixed_depth->FixedDepth = 700.0;
	config_fixed_depth->TimeStep = 0.0;
	config_fixed_depth->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_depth->VisType = VisualizeType::kFixedDepth;
	config_fixed_depth->PositionType = CalcPositionType::kPoint;



	ImageBuffer<double>* img1 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
	ImageBuffer<double>* img2 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
	ImageBuffer<double>* img3 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());

	// remapping
	float time1 = 0; // 0s
	float time2 = ONE_HOUR * 2; // 2H
	float time = ONE_HOUR;
	MPASOVisualizer::VisualizeFixedDepth(tv_mpasoField1.get(), config_fixed_depth, img1, sycl_Q);
	MPASOVisualizer::VisualizeFixedDepth(tv_mpasoField2.get(), config_fixed_depth, img2, sycl_Q);
	MPASOVisualizer::VisualizeFixedLayer_TimeVarying((int)config->imageSize.x(), (int)config->imageSize.y(), img1, img2, 0.0, 1.0, 0.75, sycl_Q);
	VTKFileManager::SaveVTI(img1, config_fixed_depth);
}

void TimeVaryingTrajectory(sycl::queue& sycl_Q, 
	std::vector<std::string>& grid_path_vec, std::vector<std::string>& vel_path_vec)
{
	
	constexpr int width = 361; constexpr int height = 181;
	VisualizationSettings* config_fixed_depth = new VisualizationSettings();
	config_fixed_depth->imageSize = vec2(width, height);
	config_fixed_depth->LatRange = vec2(20.0, 50.0);
	config_fixed_depth->LonRange = vec2(-17.0, 17.0);
	config_fixed_depth->FixedDepth = 700.0;
	config_fixed_depth->TimeStep = 0.0;
	config_fixed_depth->CalcType = CalcAttributeType::kZonalMerimoal;
	config_fixed_depth->VisType = VisualizeType::kFixedDepth;
	config_fixed_depth->PositionType = CalcPositionType::kPoint;

	SamplingSettings* sample_conf = new SamplingSettings;
	sample_conf->sampleDepth = config_fixed_depth->FixedDepth;
	sample_conf->sampleLatitudeRange = vec2(25.0, 45.0);
	sample_conf->sampleLongitudeRange = vec2(-10.0, 10.0);
	sample_conf->sampleNumer = vec2i(31, 31);

	TrajectorySettings* traj_conf = new TrajectorySettings;
	traj_conf->deltaT = ONE_MINUTE;			
	traj_conf->simulationDuration = ONE_MINUTE * 30;	
	traj_conf->recordT = ONE_MINUTE * 6;
	traj_conf->fileName = "output_line";

	tv_mpasoGrid = std::make_shared<MPASOGrid>();
	tv_mpasoGrid->initGrid(MPASOReader::loadingGridInfo(grid_path_vec[0]).get());
	tv_mpasoGrid->createKDTree("./index_timevarying.bin", sycl_Q);

	std::vector<CartesianCoord> last_points;
	std::vector<CartesianCoord> sample_points; std::vector<int> cell_id_vec;
	for (auto i = 0; i < vel_path_vec.size(); i++)
	{
		std::cout << " loading file [ " << i << " ]" << std::endl;
		traj_conf->fileName = "output_line_" + std::to_string(i);
		tv_mpasoSol1 = std::make_shared<MPASOSolution>();
		tv_mpasoSol1->initSolution(MPASOReader::loadingVelocityInfo(vel_path_vec[i], 0).get());
		tv_mpasoSol1->calcCellVertexZtop(tv_mpasoGrid.get(), sycl_Q);
		tv_mpasoSol1->calcCellCenterVelocity(tv_mpasoGrid.get(), sycl_Q);
		tv_mpasoSol1->calcCellVertexVelocity(tv_mpasoGrid.get(), sycl_Q);

		tv_mpasoField1 = std::make_shared<MPASOField>();
		tv_mpasoField1->initField(tv_mpasoGrid, tv_mpasoSol1);

		
		if (i == 0)
		{
			
			MPASOVisualizer::GenerateSamplePoint(sample_points, sample_conf);
			VTKFileManager::SavePointAsVTP(sample_points, "output_origin");
			tv_mpasoField1->calcInWhichCells(sample_points, cell_id_vec);
			std::cout << std::fixed << std::setprecision(4) << "before trajector cell_id [ " << cell_id_vec[0] << " ]" << " " << sample_points[0].x() << " " << sample_points[0].y() << " " << sample_points[0].z() << std::endl;
			last_points = MPASOVisualizer::VisualizeTrajectory(tv_mpasoField1.get(), sample_points, traj_conf, cell_id_vec, sycl_Q);
		}
		else
		{
			sample_points.clear();sample_points = last_points; cell_id_vec.clear();
			std::cout << "sample points size " << sample_points.size() << " last points size " << last_points.size() << std::endl;
			tv_mpasoField1->calcInWhichCells(sample_points, cell_id_vec);
			std::cout << std::fixed << std::setprecision(4) << "before trajector cell_id [ " << cell_id_vec[0] << " ]" << " " << sample_points[0].x() << " " << sample_points[0].y() << " " << sample_points[0].z() << std::endl;
			last_points = MPASOVisualizer::VisualizeTrajectory(tv_mpasoField1.get(), sample_points, traj_conf, cell_id_vec, sycl_Q);
		}


	}
	

	
	
	//std::cout << "points size() " << sample_points.size() << std::endl;
	
	
	
	

	
}


bool parseCommandLine(int argc, char* argv[], std::string& input_yaml_filename, std::string& data_path_prefix) 
{
    cxxopts::Options options(argv[0]);
	options.add_options()
		("input,i", "Input yaml file", cxxopts::value<std::string>(input_yaml_filename))
		("prefix,p", "Data path prefix", cxxopts::value<std::string>(data_path_prefix))
		("help,h", "Print this information");

	auto results = options.parse(argc, argv);
	if (results.count("help")) 
	{
        std::cout << options.help() << std::endl;
        return false;
    }
	if (!results.count("input")) 
	{
        Debug("[ERROR]::No input file detected");
        return false;
    }
	return true;
}

int main(int argc, char* argv[])
{

	if (!parseCommandLine(argc, argv, input_yaml_filename, data_path_prefix)) {
        exit(1);
    }
	
    path = input_yaml_filename.c_str();

	sycl::queue sycl_Q;
#if __linux__
	sycl_Q = sycl::queue(sycl::gpu_selector_v);
#elif _WIN32
	sycl_Q = sycl::queue(sycl::cpu_selector_v);
#elif __APPLE__
	sycl_Q = nullptr;
#endif
	std::cout << " [ system information" << " ]\n";
	std::cout << "Device selected : " << sycl_Q.get_device().get_info<sycl::info::device::name>() << "\n";
	std::cout << "Device vendor   : " << sycl_Q.get_device().get_info<sycl::info::device::vendor>() << "\n";
	std::cout << "Device version  : " << sycl_Q.get_device().get_info<sycl::info::device::version>() << "\n";

	std::shared_ptr<ftk::stream> stream(new ftk::stream);
	if (!data_path_prefix.empty()) stream->set_path_prefix(data_path_prefix);
	std::cout << " [ files information" << " ]\n";
	stream->parse_yaml(input_yaml_filename);
	

	TimeFixed(sycl_Q, stream.get());	

	// std::vector<std::string> grid_path_vec, vel_path_vec;
	// grid_path_vec.push_back("../soma/output_1.nc");	vel_path_vec.push_back("../soma/output_1.nc");
	// grid_path_vec.push_back("../soma/output_2.nc");	vel_path_vec.push_back("../soma/output_2.nc");
	// grid_path_vec.push_back("../soma/output_3.nc");	vel_path_vec.push_back("../soma/output_3.nc");
	// grid_path_vec.push_back("../soma/output_4.nc");	vel_path_vec.push_back("../soma/output_4.nc");
	// grid_path_vec.push_back("../soma/output_5.nc");	vel_path_vec.push_back("../soma/output_5.nc");



	//TimeVarying(sycl_Q, grid_path_vec, grid_path_vec);
	// TimeVaryingTrajectory(sycl_Q, grid_path_vec, grid_path_vec);

	system("pause");
	return 0;
}
