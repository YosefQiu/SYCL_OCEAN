#pragma once
#include "ggl.h"
#include "Log.hpp"

class MPASOReader
{
public:
	using Ptr = std::shared_ptr<MPASOReader>;
	MPASOReader() = default;
	MPASOReader(const std::string& path);
	~MPASOReader() {}
public:
	int mCellsSize;
	int mEdgesSize;
	int mMaxEdgesSize;
	int mVertexSize;

	int currentTimestep;
	int mTimesteps;
	int mVertLevels;
	int mVertLevelsP1;
public:
	std::vector<vec3>		vertexCoord_vec;
	std::vector<vec3>		cellCoord_vec;
	std::vector<vec3>		edgeCoord_vec;
	std::vector<vec2>		vertexLatLon_vec;

	std::vector<size_t>		verticesOnCell_vec;
	std::vector<size_t>		cellsOnVertex_vec;
	std::vector<size_t>		cellsOnCell_vec;
	std::vector<size_t>		numberVertexOnCell_vec;
	std::vector<size_t>     cellsOnEdge_vec;
	std::vector<size_t>     edgesOnCell_vec;

	// related velocity --> need to split a another class.
	std::vector<vec3>		cellVelocity_vec;        // nCellsSize * nVertLevels;
	std::vector<double>	    cellLayerThickness_vec;  // nCellsSize * nVertLevels
	std::vector<double>     cellZTop_vec;            // nCellsSize * nVertLevels
	std::vector<double>     cellVertVelocity_vec;    // nCellsSize * nVertLevelsP1
	std::vector<double>     cellNormalVelocity_vec;  // nEdges * nVertLevels
	std::vector<double>     cellMeridionalVelocity_vec; // nCellsSize for surface
	std::vector<double>	 	cellZonalVelocity_vec;      // nCellsSize for surface
	std::vector<double>     cellBottomDepth_vec; 	   // nCellsSize

public:
	static Ptr loadingGridInfo(const std::string& path)
	{
		std::shared_ptr<MPASOReader> reader(new MPASOReader(path));
		reader->loadingData(); 
		return reader;
	}
	static Ptr loadingVelocityInfo(const std::string& path, const int& timestep)
	{
		std::shared_ptr<MPASOReader> reader(new MPASOReader(path));
		reader->loadingVelocity(timestep);
		return reader;
	}
	void loadingData();
	void loadingVelocity(const int& timestep);
private:
	void loadingInt(const int size, std::vector<size_t>& data, std::string xValue);
	void loadingDouble(const int size, std::vector<double>& data, std::string xValue);
	void loadingVec2(const int size, std::vector<vec2>& data, std::string xValue, std::string yValue);
	void loadingVec3(const int size, std::vector<vec3>& data, std::string xValue, std::string yValue, std::string zVale);
	
	void ReadNormalVelocity(const int timestep, std::vector<double>& data);
	void ReadMeridionalVelocity(const int timestep, std::vector<double>& data);
	void ReadZonalVelocity(const int timestep, std::vector<double>& data);
	void ReadVelocity(const int timestep, std::vector<vec3>& data);
	void ReadVertVelocityTop(const int timestep, std::vector<double>& data);
	void ReadLayerThickness(const int timestep, std::vector<double>& data);
	void ReadZTop(const int timestep, std::vector<double>& data);

private:
	std::string path;
	//netCDF::NcFile* ncFile;
	std::unique_ptr<netCDF::NcFile> ncFile;
};

