
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

std::shared_ptr<MPASOGrid> mpasoGrid = nullptr;
std::shared_ptr<MPASOSolution> mpasoSol = nullptr;
std::shared_ptr<MPASOField> mpasoField = nullptr;


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

int main(int argc, char* argv[])
{


	cxxopts::Options options(argv[0]);
	options.add_options()
		("input,i", "Input yaml file", cxxopts::value<std::string>(input_yaml_filename))
		("prefix,p", "Data path prefix", cxxopts::value<std::string>(data_path_prefix))
		("help,h", "Print this information");

	options.parse_positional({ "input" });
	auto results = options.parse(argc, argv);

	if (!results.count("input") || results.count("help")) {
#if _WIN32
		input_yaml_filename = "../mpas.yaml";
#elif __linux__
		// path = path = "../SOMA_output.nc";
		input_yaml_filename = "../mpas.yaml";
		exit(0);
#endif
	}
	std::cout << " [ " << input_yaml_filename << " ]\n";
	path = input_yaml_filename.c_str();

	sycl::queue sycl_Q;
#if __linux__
	sycl_Q = sycl::queue(sycl::gpu_selector_v);
#elif _WIN32
	sycl_Q = sycl::queue(sycl::cpu_selector_v);
#elif __APPLE__
	sycl_Q = nullptr;
#endif
	std::cout << "Device selected : " << sycl_Q.get_device().get_info<sycl::info::device::name>() << "\n";
	std::cout << "Device vendor   : " << sycl_Q.get_device().get_info<sycl::info::device::vendor>() << "\n";
	std::cout << "Device version  : " << sycl_Q.get_device().get_info<sycl::info::device::version>() << "\n";

	std::shared_ptr<ftk::stream> stream(new ftk::stream);
	if (!data_path_prefix.empty()) stream->set_path_prefix(data_path_prefix);

	stream->parse_yaml(input_yaml_filename);
	auto grid_path = stream->substreams[0]->filenames[0];
	auto vel_path = stream->substreams[1]->filenames[0];
	
	auto gs = stream->read_static();
	gs->print_info(std::cerr);

	for (int i = 0; i < stream->total_timesteps(); i++) {
		auto g = stream->read(i);
		g->print_info(std::cerr);
	}

	mpasoGrid = std::make_shared<MPASOGrid>();
	mpasoGrid->initGrid(MPASOReader::loadingGridInfo(grid_path).get());
	mpasoGrid->createKDTree("./index.bin", sycl_Q);

	mpasoSol = std::make_shared<MPASOSolution>();
	mpasoSol->initSolution(MPASOReader::loadingVelocityInfo(vel_path, 0).get());
	//mpasoSol->calcCellCenterZtop();
	mpasoSol->calcCellVertexZtop(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellCenterVelocity(mpasoGrid.get(), sycl_Q);
	mpasoSol->calcCellVertexVelocity(mpasoGrid.get(), sycl_Q);

	mpasoField = std::make_shared<MPASOField>();
	mpasoField->initField(mpasoGrid, mpasoSol);


#if __linux__
	sycl::queue SYCL_Q = sycl::queue(sycl::gpu_selector_v);
#elif _WIN32
	sycl::queue SYCL_Q = sycl::queue(sycl::cpu_selector_v);
#endif

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



	ImageBuffer<double>* img1 = new ImageBuffer<double>(config->imageSize.x(), config->imageSize.y());
	ImageBuffer<double>* img2 = new ImageBuffer<double>(config_fixed_depth->imageSize.x(), config_fixed_depth->imageSize.y());
	ImageBuffer<double>* img3 = new ImageBuffer<double>(config_fixed_lat->imageSize.x(), config_fixed_lat->imageSize.y());

	// remapping
	MPASOVisualizer::VisualizeFixedLayer(mpasoField.get(), config, img1, sycl_Q);
	VTKFileManager::SaveVTI(img1, config);
	MPASOVisualizer::VisualizeFixedDepth(mpasoField.get(), config_fixed_depth, img2, sycl_Q);
	VTKFileManager::SaveVTI(img2, config_fixed_depth);
	MPASOVisualizer::VisualizeFixedLatitude(mpasoField.get(), config_fixed_lat, img3, sycl_Q);
  Debug("finished... fixed_lat_35.000000.vti");


	
	SamplingSettings* sample_conf = new SamplingSettings;
	sample_conf->sampleDepth = config_fixed_depth->FixedDepth;
	sample_conf->sampleLatitudeRange = vec2(25.0, 45.0);
	sample_conf->sampleLongitudeRange = vec2(-10.0, 10.0);
	sample_conf->sampleNumer = vec2i(31, 31);

	TrajectorySettings* traj_conf = new TrajectorySettings;
	traj_conf->deltaT = ONE_HOUR;			
	traj_conf->simulationDuration = ONE_YEAR * 20;	
	traj_conf->recordT = ONE_DAY * 2;
	traj_conf->fileName = "output_line";

	std::vector<CartesianCoord> sample_points; std::vector<int> cell_id_vec;
	MPASOVisualizer::GenerateSamplePoint(sample_points, sample_conf);
	//std::cout << "points size() " << sample_points.size() << std::endl;
	VTKFileManager::SavePointAsVTP(sample_points, "output_origin");
	
	mpasoField->calcInWhichCells(sample_points, cell_id_vec);
	std::cout << std::fixed << std::setprecision(4) << "before trajector cell_id [ " << cell_id_vec[0] << " ]" << " " << sample_points[0].x() << " " << sample_points[0].y() << " " << sample_points[0].z() << std::endl;

	MPASOVisualizer::VisualizeTrajectory(mpasoField.get(), sample_points, traj_conf, cell_id_vec, sycl_Q);

	

	system("pause");
	return 0;
}
