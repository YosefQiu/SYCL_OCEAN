#include "MPASOSolution.h"
#include "Interpolation.hpp"
void MPASOSolution::initSolution(MPASOReader* reader)
{
    this->mCurrentTime = std::move(reader->currentTimestep);
    this->mCellsSize = reader->mCellsSize;
    this->mEdgesSize = reader->mEdgesSize;
    this->mMaxEdgesSize = reader->mMaxEdgesSize;
    this->mVertexSize = reader->mVertexSize;
    this->mTimesteps = reader->mTimesteps;
    this->mVertLevels = reader->mVertLevels;
    this->mVertLevelsP1 = reader->mVertLevelsP1;

    this->cellVelocity_vec = std::move(reader->cellVelocity_vec);
    this->cellLayerThickness_vec = std::move(reader->cellLayerThickness_vec);
    this->cellZTop_vec = std::move(reader->cellZTop_vec);
    this->cellVertVelocity_vec = std::move(reader->cellVertVelocity_vec);
    this->cellNormalVelocity_vec = std::move(reader->cellNormalVelocity_vec);
    this->cellMeridionalVelocity_vec = std::move(reader->cellMeridionalVelocity_vec);
    this->cellZonalVelocity_vec = std::move(reader->cellZonalVelocity_vec);
    this->cellBottomDepth_vec = std::move(reader->cellBottomDepth_vec);

    // calcCellCenterZtop();
}

void MPASOSolution::getCellVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    vel = cell_on_velocity[idx];
}

void MPASOSolution::getCellVertVelocity(const size_t cell_id,
    const size_t level,
    std::vector<double>& cell_vert_velocity,
    double& vel)
{
    auto VertLevelsP1 = mVertLevelsP1;
    if (VertLevelsP1 == -1 || VertLevelsP1 == 0)
    {
        Debug("ERROR, VertLevelsP1 is not defined");
    }
    auto idx = VertLevelsP1 * cell_id + level;
    vel = cell_vert_velocity[idx];
}

void MPASOSolution::getCellZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop)
{
    // 单元格0，层0的顶部Z坐标
    // 单元格0，层1的顶部Z坐标
    // 单元格0，层2的顶部Z坐标
    // 单元格1，层0的顶部Z坐标
    // 单元格1，层1的顶部Z坐标
    // 单元格1，层2的顶部Z坐标
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    ztop = cell_ztop[idx];
}

void MPASOSolution::getEdgeNormalVelocity(const size_t edge_id, const size_t level, std::vector<double>& edge_normal_velocity, double& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * edge_id + level;
    vel = edge_normal_velocity[idx];
}

void MPASOSolution::getCellLayerThickness(const size_t cell_id, const size_t level, std::vector<double>& cell_thickness, double& thinckness)
{
    // 单元格0，层0的厚度
    // 单元格0，层1的厚度
    // 单元格0，层2的厚度
    // 单元格1，层0的厚度
    // 单元格1，层1的厚度
    // 单元格1，层2的厚度

    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    thinckness = cell_thickness[idx];
}

void MPASOSolution::getCellSurfaceMeridionalVelocity(const size_t cell_id, std::vector<double>& cell_meridional_velocity, double& vel)
{
    vel = cell_meridional_velocity[cell_id];
}

void MPASOSolution::getCellSurfaceZonalVelocity(const size_t cell_id, std::vector<double>& cell_zonal_velocity, double& vel)
{
    vel = cell_zonal_velocity[cell_id];
}

void MPASOSolution::getCellCenterZTop(const size_t cell_id, const size_t level, std::vector<double>& cell_ztop, double& ztop)
{
    // 单元格0，层0的顶部Z坐标
    // 单元格0，层1的顶部Z坐标
    // 单元格0，层2的顶部Z坐标
    // 单元格1，层0的顶部Z坐标
    // 单元格1，层1的顶部Z坐标
    // 单元格1，层2的顶部Z坐标
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    ztop = cell_ztop[idx];
}

void MPASOSolution::getCellVertexZTop(const size_t vertex_id, const size_t level, std::vector<double>& cell_vertex_ztop, double& ztop)
{
    ztop = cell_vertex_ztop[vertex_id * mVertLevels + level];
}

void MPASOSolution::getCellCenterVelocity(const size_t cell_id, const size_t level, std::vector<vec3>& cell_on_velocity, vec3& vel)
{
    auto VertLevels = mVertLevels;
    if (VertLevels == -1 || VertLevels == 0)
    {
        Debug("ERROR, VertLevels is not defined");
    }
    auto idx = VertLevels * cell_id + level;
    vel = cell_on_velocity[idx];
}

void MPASOSolution::getCellVertexVelocity(const size_t vertex_id, const size_t level, std::vector<vec3>& cell_vertex_velocity, vec3& vel)
{
    vel = cell_vertex_velocity[vertex_id * mVertLevels + level];
}




void MPASOSolution::calcCellCenterZtop()
{
    Debug(("[MPASOSolution]::Calc Cell Center Z Top at t = " + std::to_string(mCurrentTime)).c_str());

    // 1. 判断是否有layerThickness
    if (cellLayerThickness_vec.empty())
    {
        Debug("ERROR, cellLayerThickness is not defined");
        exit(0);
    }

    // 2. 计算ZTop
    auto nCellsSize = mCellsSize;
    //auto nVertLevelsP1      =  mVertLevelsP1;
    auto nVertLevels = mVertLevels;
    auto nTimesteps = mCurrentTime;

    //std::cout << "=======\n";
    //std::cout << "nCellsSize = " << nCellsSize << std::endl;
    ////std::cout << "nVertLevelsP1 = " << nVertLevelsP1 << std::endl;
    //std::cout << "nVertLevels = " << nVertLevels << std::endl;
    //std::cout << "nTimesteps = " << nTimesteps << std::endl;

    if (!cellZTop_vec.empty()) cellZTop_vec.clear();
    cellZTop_vec.resize(nCellsSize * nVertLevels);

    for (size_t i = 0; i < nCellsSize; ++i)
    {
        // 初始化最顶层
        cellZTop_vec[i * nVertLevels] = 0.0;
        for (size_t j = 1; j < nVertLevels; ++j)
        {
            double layerThickness;
            getCellLayerThickness(i, j - 1, cellLayerThickness_vec, layerThickness);
            cellZTop_vec[i * nVertLevels + j] = cellZTop_vec[i * nVertLevels + j - 1] - layerThickness;
        }
    }



    for (auto i = 0; i < cellZTop_vec.size(); ++i)
    {
        cellZTop_vec[i] *= 1.0;
    }

    mTotalZTopLayer = nVertLevels;

    std::cout << "cellZTop_vec.size() = " << cellZTop_vec.size() << std::endl;
    std::cout << "nCellSize x nVertLevels = " << nCellsSize * nVertLevels << std::endl;
}



template<typename T>
void writeVertexZTopToFile(const std::vector<T>& vertexZTop_vec, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile) {
        size_t size = vertexZTop_vec.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write the size of the vector
        outFile.write(reinterpret_cast<const char*>(vertexZTop_vec.data()), size * sizeof(T)); // Write the data
        std::cout << "Wrote " << size << " elements to " << filename << std::endl;
        outFile.close();
    }
    else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}

template<typename T>
void readVertexZTopFromFile(std::vector<T>& vertexZTop_vec, const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (inFile) {
        size_t size;
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size)); // Read the size of the vector
        vertexZTop_vec.resize(size);
        inFile.read(reinterpret_cast<char*>(vertexZTop_vec.data()), size * sizeof(T)); // Read the data
        inFile.close();
        std::cout << "Read " << size << " elements from " << filename << std::endl;
    }
    else {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
    }
}


void saveDataToTextFile2(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (const auto& value : data) {
        outfile << value.x() << " " << value.y() << " " << value.x() << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to " << filename << std::endl;
}

//TODO
void saveDataToTextFile(const std::vector<double>& data, const std::string& filename) {
    int VERTEX_NUM = 6;
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    int count = 0;
    for (const auto& value : data) {
        outfile << value << " ";
        count++;
        if (count % VERTEX_NUM == 0) {
            outfile << std::endl;
        }
    }

    if (count % VERTEX_NUM != 0) {  // 如果数据总数不是7的倍数，在文件末尾添加换行
        outfile << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to " << filename << std::endl;
}


void saveDataToTextFile3(const std::vector<vec3>& data, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    int VERTEX_NUM = 6;
    int count = 0;
    for (const auto& value : data) {
        outfile << value.x() << " ";
        count++;
        if (count % VERTEX_NUM == 0) {
            outfile << std::endl;
        }
    }

    if (count % VERTEX_NUM != 0) {  // 如果数据总数不是7的倍数，在文件末尾添加换行
        outfile << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to " << filename << std::endl;
}



void MPASOSolution::calcCellVertexZtop(MPASOGrid* grid, sycl::queue& q)
{
    Debug(("[MPASOSolution]::Calc Cell Vertex Z Top at t = " + std::to_string(mCurrentTime)).c_str());
    
    if (mTotalZTopLayer == 0 || mTotalZTopLayer == -1) mTotalZTopLayer = mVertLevels;
    if (!cellVertexZTop_vec.empty()) cellVertexZTop_vec.clear();
    cellVertexZTop_vec.resize(grid->vertexCoord_vec.size() * mTotalZTopLayer);

#if _WIN32 || __APPLE__
    if (std::filesystem::exists("cellVertexZTop_vec.bin")) {
        readVertexZTopFromFile<double>(cellVertexZTop_vec, "cellVertexZTop_vec.bin");
        return;
    }
#endif
    
#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL 中心坐标

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<double, 1> cellCenterZTop_buf(cellZTop_vec.data(), sycl::range<1>(cellZTop_vec.size()));                           //CELL 中心ZTOP
    sycl::buffer<double, 1> cellVertexZTop_buf(cellVertexZTop_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size() * mTotalZTopLayer));        //CELL 顶点ZTOP （要求的）
  

    q.submit([&](sycl::handler& cgh) {
        
        auto acc_vertexCoord_buf        = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf          = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf     = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf      = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterZTop_buf     = cellCenterZTop_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexZTop_buf     = cellVertexZTop_buf.get_access<sycl::access::mode::read_write>(cgh);
        
        
        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1]; 

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = 8521;
            const int VERTEX_NUM = 6;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = 60;
            const int VERTLEVELS = 60;
            // 1. 找出这个cell 的所有顶点 不存在的置为nan
            
            // 1.1 计算这个CELL 有多少个顶点
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            // 1.2 找出所有候选顶点
            size_t current_cell_vertices_idx[VERTEX_NUM];
            for (size_t k = 0; k < VERTEX_NUM; ++k)
            {
                current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
            }
            // 1.3 不存在的顶点设置为nan
            auto nan = std::numeric_limits<size_t>::max();
            for (size_t k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
            {
                current_cell_vertices_idx[k] = nan;
            }
            // =============================== 找到7个顶点

            double current_cell_vertices_value[VERTEX_NUM];
            bool bBoundary = false;
            for (auto k = 0; k < VERTEX_NUM; ++k)
            {
                auto vertex_idx = current_cell_vertices_idx[k];
                // 2.1 如果是nan 跳过
                if (vertex_idx == nan) { current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN(); continue; }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                // 2.2 如果不是nan 找到含有这个顶点的3个cell id(候选) 边界情况没有3个
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                // 2.3 找到这3个CELL 的中心ZTOP
                double tmp_cell_center_ztop[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    double value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = 0.0;
                        tmp_cell_center_ztop[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        double ztop;
                        auto ztop_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        ztop = acc_cellCenterZTop_buf[ztop_idx];
                        tmp_cell_center_ztop[tmp_cell] = ztop;
                    }
                }
                //current_cell_vertices_value[k] = 1.0 * tmp_cell_center_ztop[0] + 1.0 * tmp_cell_center_ztop[1] + 1.0 * tmp_cell_center_ztop[2];


                // 2.4 如果是边界点
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_ztop[0] + 0.0 * tmp_cell_center_ztop[1] + 0.0 * tmp_cell_center_ztop[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_ztop[0] + v * tmp_cell_center_ztop[1] + w * tmp_cell_center_ztop[2];
                }

                acc_cellVertexZTop_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }
        


        });
    });
    q.wait_and_throw();
    Debug("finished the sycl part");
    auto host_accessor = cellVertexZTop_buf.get_access<sycl::access::mode::read>();
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // 获取缓冲区的总大小

    std::cout << "acc_cellVertexZTop_buf.size() = " << cellVertexZTop_vec.size() << " " << acc_length << std::endl;
    std::cout << "mVerticesSize x nVertLevels = " << grid->mVertexSize * mVertLevels << std::endl;
    //saveDataToTextFile(cellVertexZTop_vec, "OUTPUT1_ztop.txt");
    writeVertexZTopToFile<double>(cellVertexZTop_vec, "cellVertexZTop_vec.bin");
    Debug("finish Calc cellVertexZTop_vec");
#endif

#if USE_SYCL == 0
    cellVertexZTop_vec.clear();

    for (auto j = 0; j < mCellsSize; ++j)
    {
        for (auto i = 0; i < mTotalZTopLayer; ++i)
        {
            auto cell_id = j;
            auto current_layer = i;
            // 1.1 找出这个cell 的所有顶点
            std::vector<size_t> current_cell_vertices_idx;
            // 1.2 计算这个cell 有多少个顶点
            auto current_cell_vertices_number = grid->numberVertexOnCell_vec[cell_id];
            grid->getVerticesOnCell(cell_id, grid->verticesOnCell_vec, current_cell_vertices_idx);
            // 1.3 不存在的顶点设置为nan
            auto nan = std::numeric_limits<size_t>::max();
            if (current_cell_vertices_idx.size() != current_cell_vertices_number)
            {
                auto length = current_cell_vertices_idx.size() - current_cell_vertices_number;
                current_cell_vertices_idx.resize(current_cell_vertices_number);
                current_cell_vertices_idx.insert(current_cell_vertices_idx.end(), length, nan);
            }
            //=========================== 找到7个顶点 V

            


            std::vector<double> current_cell_vertices_value;
            if (!current_cell_vertices_value.empty()) current_cell_vertices_value.clear();
            current_cell_vertices_value.resize(current_cell_vertices_idx.size());

            // 遍历7个顶点来计算7个顶点的ZTOP 存在current_cell_vertices_value
            bool bBoundary = false;
            for (auto k = 0; k < current_cell_vertices_idx.size(); ++k)
            {
                // 如果是nan 值 就跳过
                if (current_cell_vertices_idx[k] == nan) { current_cell_vertices_value[k] = std::numeric_limits<double>::quiet_NaN(); continue; };

                // 找出这个顶点的坐标
                auto vertex_idx = current_cell_vertices_idx[k];
                auto current_vertex = grid->vertexCoord_vec[vertex_idx];
                // 找出这个顶点的所有CELL 预计3个
                std::vector<size_t> tmp_cell_id;
                grid->getCellsOnVertex(vertex_idx, grid->cellsOnVertex_vec, tmp_cell_id);

                std::vector<double> tmp_cell_center_ztop; // 用来存相邻的这个3个CELL的中心ZTOP
                if (!tmp_cell_center_ztop.empty()) tmp_cell_center_ztop.clear();
                tmp_cell_center_ztop.resize(3);
                // 遍历这3个CELL，分别找出存在中心的CELL的ZTOP
                for (auto tmp_cell = 0; tmp_cell < 3; tmp_cell++)
                {
                    double value;
                    // 这个CELL是边界CELL，不存在3个邻接CELL
                    if (tmp_cell_id[tmp_cell] > grid->mCellsSize + 1)
                    {
                        value = double(0.0f);
                        tmp_cell_center_ztop[tmp_cell] = value;
                        bBoundary = true;
                    }
                    // 这个CELL 存在3个相邻的CELL
                    else
                    {
                        // 获取这个CELL在第K（current_layer）层的中心的ZTOP
                        double ztop;
                        getCellZTop(tmp_cell_id[tmp_cell], current_layer, cellZTop_vec, ztop);
                        tmp_cell_center_ztop[tmp_cell] = ztop;
                    }
                }

                // 如果不是边界点
                if (!bBoundary)
                {
                    float u, v, w;
                    Interpolator::TRIANGLE tri(grid->cellCoord_vec[tmp_cell_id[0]], grid->cellCoord_vec[tmp_cell_id[1]], grid->cellCoord_vec[tmp_cell_id[2]]);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);

                    current_cell_vertices_value[k] = u * tmp_cell_center_ztop[0] + v * tmp_cell_center_ztop[1] + w * tmp_cell_center_ztop[2];
                }
                else
                {
                    // 边界点
                    current_cell_vertices_value[k] = 0.0f * tmp_cell_center_ztop[0] + 0.0f * tmp_cell_center_ztop[1] + 0.0f * tmp_cell_center_ztop[2];
                }
            
            }

            //=========================== 找到7个顶点的ZTOP V

            // 存到cellVertexZTop_vec
            for (auto k = 0; k < current_cell_vertices_value.size(); ++k)
            {
                cellVertexZTop_vec.push_back(current_cell_vertices_value[k]);
            }

            
        }
    }
    
    std::cout << "cellVertexZTop_vec.size() = " << cellVertexZTop_vec.size() << std::endl;
    writeVertexZTopToFile<double>(cellVertexZTop_vec, "cellVertexZTop_vec.bin");
    Debug("finish2");
#endif
    
}

void MPASOSolution::calcCellCenterVelocity(MPASOGrid* grid, sycl::queue& q)
{

    Debug(("[MPASOSolution]::Calc Center Velocity at t = " + std::to_string(mCurrentTime)).c_str());

    if (!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(mCellsSize * mTotalZTopLayer);

#if _WIN32 || __APPLE__
    if (std::filesystem::exists("cellCenterVelocity_vec.bin")) {
        readVertexZTopFromFile<vec3>(cellCenterVelocity_vec, "cellCenterVelocity_vec.bin");
        return;
    }
#endif

#if USE_SYCL
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL 中心坐标
    sycl::buffer<vec3, 1> edgeCoord_buf(grid->edgeCoord_vec.data(), sycl::range<1>(grid->edgeCoord_vec.size()));       // EDGE 中心坐标

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> edgesOnCell_buf(grid->edgesOnCell_vec.data(), sycl::range<1>(grid->edgesOnCell_vec.size()));                       // CELL 边ID
    sycl::buffer<size_t, 1> cellsOnEdge_buf(grid->cellsOnEdge_vec.data(), sycl::range<1>(grid->cellsOnEdge_vec.size()));                       // EDGE 细胞ID

    sycl::buffer<double, 1> cellNormalVelocity_buf(cellNormalVelocity_vec.data(), sycl::range<1>(cellNormalVelocity_vec.size()));             // EDGE 正常速度
    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cellCenterVelocity_vec.data(), sycl::range<1>(cellCenterVelocity_vec.size()));        // CELL 中心速度

    q.submit([&](sycl::handler& cgh) {

        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgeCoord_buf = edgeCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_edgesOnCell_buf = edgesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnEdge_buf = cellsOnEdge_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellNormalVelocity_buf = cellNormalVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = 8521;
            const int VERTEX_NUM = 6;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = 60;
            const int VERTLEVELS = 60;
            // 1. 根据cell_id 找到cell position
            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            vec3 cell_position = acc_cellCoord_buf[cell_id];
            // 2. 根据cell_id 找到所有edge_id(候选7个)
            size_t current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            size_t current_cell_edges_id[VERTEX_NUM];
            for (auto k = 0; k < current_cell_vertices_number; ++k)
            {
                current_cell_edges_id[k] = acc_edgesOnCell_buf[cell_id * VERTEX_NUM + k] -1;
            }
            // 2.1 将不存在的edge_id 设置为nan
            auto nan = std::numeric_limits<size_t>::max();
            for (auto k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
            {
                current_cell_edges_id[k] = nan;
            }
            // =============================== 找到7个边
            // 3. 遍历 -》 计算每条边的矢量速度
            for (auto k = 0; k < VERTEX_NUM; ++k)
            {
                auto edge_id = current_cell_edges_id[k];
                if (edge_id == nan) { continue; } // TODO 考虑最后除法是否会有多计算一次
                // 3.1 找到变得中心位置
                vec3 edge_position = acc_edgeCoord_buf[edge_id];
                // 3.2 找到边的 normal vector
                size_t tmp_cell_id[2];
                tmp_cell_id[0] = acc_cellsOnEdge_buf[edge_id * 2.0f + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnEdge_buf[edge_id * 2.0f + 1] - 1;
                auto min_cell_id = tmp_cell_id[0] < tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                auto max_cell_id = tmp_cell_id[0] > tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                vec3 normal_vector;
                if (max_cell_id > CELL_SIZE)
                {
                    vec3 edge_position_min = acc_edgeCoord_buf[min_cell_id];
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    normal_vector = edge_position_min - min_cell_position;
                    double length = YOSEF_LENGTH(normal_vector);
                    normal_vector /= length;
                }
                else
                {
                    vec3 min_cell_position = acc_cellCoord_buf[min_cell_id];
                    vec3 max_cell_position = acc_cellCoord_buf[max_cell_id];
                    // 3.2 方法一 两个cell的中心位置的差
                    normal_vector = max_cell_position - min_cell_position;
                    double length = YOSEF_LENGTH(normal_vector);
                    normal_vector /= length;
                }
                // 3.3 找到边的 normal velocity
                auto normal_velocity = acc_cellNormalVelocity_buf[edge_id * TOTAY_ZTOP_LAYER + current_layer];
                vec3 current_edge_velocity;
                current_edge_velocity.x() = normal_velocity * normal_vector.x();
                current_edge_velocity.y() = normal_velocity * normal_vector.y();
                current_edge_velocity.z() = normal_velocity * normal_vector.z();
                cell_center_velocity += current_edge_velocity;
            }
            // 4. 平均
            cell_center_velocity /= static_cast<double>(current_cell_vertices_number);
            cell_center_velocity *= 2.0;
            acc_cellCenterVelocity_buf[cell_id * TOTAY_ZTOP_LAYER + current_layer] = cell_center_velocity;

        });
    });
    q.wait_and_throw();

    Debug("finished the sycl part");
    auto host_accessor = cellCenterVelocity_buf.get_access<sycl::access::mode::read>();
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // 获取缓冲区的总大小

    std::cout << "cellCenterVelocity_buf.size() = " << cellCenterVelocity_vec.size() << " " << acc_length << std::endl;
    std::cout << "mCellSize x nVertLevels = " << grid->mCellsSize * mVertLevels << std::endl;
    //saveDataToTextFile2(cellCenterVelocity_vec, "OUTPUT1_ztop.txt");
    writeVertexZTopToFile<vec3>(cellCenterVelocity_vec, "cellCenterVelocity_vec.bin");
    Debug("finish Calc cellCenterVelocity_vec");
#endif

#if USE_SYCL == 0
    if(!cellCenterVelocity_vec.empty()) cellCenterVelocity_vec.clear();
    cellCenterVelocity_vec.resize(grid->mCellsSize * 60);



    for (auto j = 0; j < mCellsSize; ++j)
    {
        for (auto i = 0; i < mTotalZTopLayer; ++i)
        {
            auto cell_id = j;
            auto current_layer = i;

            vec3 cell_center_velocity = { 0.0, 0.0, 0.0 };
            // 1. 根据cell_id 找到 cell position
            vec3f cell_position = grid->cellCoord_vec[cell_id];
            // 2. 根据cell id 找到所有的edge id
            std::vector<size_t> current_cell_edges_id;
            if (!current_cell_edges_id.empty()) current_cell_edges_id.clear();
            grid->getEdgesOnCell(cell_id, grid->edgesOnCell_vec, current_cell_edges_id);
            auto current_cell_vertices_number = grid->numberVertexOnCell_vec[cell_id];
            // 2.1 删除不存在的边ID
            if (current_cell_edges_id.size() != current_cell_vertices_number)
            {
                current_cell_edges_id.resize(current_cell_vertices_number);
            }
            // 3. 遍历 -》 计算每条边的矢量速度
            std::vector<vec3f> current_cell_velocity_vec;
            std::vector<vec3f> current_cell_edge_position_vec;
            for (auto k = 0; k < current_cell_edges_id.size(); ++k)
            {
                auto edge_id = current_cell_edges_id[k];
                // 3.1 找到边的中心位置坐标
                vec3f edge_position;
                if (edge_id < grid->edgeCoord_vec.size())
                    edge_position = grid->edgeCoord_vec[edge_id];
                else
                    std::cout << "error\n";
                current_cell_edge_position_vec.push_back(edge_position);
                // 3.2 找到边的normal vector
                std::vector<size_t> tmp_cell_id;
                vec3f normal_vector;
                grid->getCellsOnEdge(edge_id, grid->cellsOnEdge_vec, tmp_cell_id);
                auto min_cell_id = tmp_cell_id[0] < tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                auto max_cell_id = tmp_cell_id[0] > tmp_cell_id[1] ? tmp_cell_id[0] : tmp_cell_id[1];
                if (max_cell_id > grid->mCellsSize)
                {
                    vec3f min_cell_position = grid->cellCoord_vec[min_cell_id];
                    normal_vector = min_cell_position;
                    float length = YOSEF_LENGTH(normal_vector);
                    normal_vector /= length;
                }
                else
                {
                    vec3f min_cell_position = grid->cellCoord_vec[min_cell_id];
                    vec3f max_cell_position = grid->cellCoord_vec[max_cell_id];
                    // 3.2 方法一 两个cell的中心位置的差
                    normal_vector = max_cell_position - min_cell_position;
                    float length = YOSEF_LENGTH(normal_vector);
                    normal_vector /= length;
                }
                // 3.3 找到 边的normal velocity
                double normal_velocity;
                getEdgeNormalVelocity(edge_id, current_layer, cellNormalVelocity_vec, normal_velocity);
                // 3.4 计算每条边的矢量速度
                vec3f current_edge_velocity;
                current_edge_velocity.x() = normal_velocity * normal_vector.x();
                current_edge_velocity.y() = normal_velocity * normal_vector.y();
                current_edge_velocity.z() = normal_velocity * normal_vector.z();
                current_cell_velocity_vec.push_back(current_edge_velocity);
                cell_center_velocity += current_edge_velocity;
            }

            // 4. 平均
            cell_center_velocity /= current_cell_edges_id.size();
            cell_center_velocity *= 2.0f;

            cellCenterVelocity_vec[cell_id * mTotalZTopLayer + current_layer] = cell_center_velocity;
        }
    }
    Debug("Calc Center Velocity Done");
    std::cout << "cellCenterVelocity_vec.size() = " << cellCenterVelocity_vec.size() << " " << grid->mCellsSize * mTotalZTopLayer << std::endl;
    writeVertexZTopToFile<vec3f>(cellCenterVelocity_vec, "cellCenterVelocity_vec.bin");
    //saveDataToTextFile2(cellCenterVelocity_vec, "OUTPUT2_ztop.txt");
#endif
}

void MPASOSolution::calcCellVertexVelocity(MPASOGrid* grid, sycl::queue& q)
{
    Debug(("[MPASOSolution]::Calc Cell Vertex Velocity at t = " + std::to_string(mCurrentTime)).c_str());

    if(!cellVertexVelocity_vec.empty()) cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(mVertexSize * mTotalZTopLayer);

#if _WIN32 || __APPLE__
    if (std::filesystem::exists("cellVertexVelocity_vec.bin")) {
        readVertexZTopFromFile<vec3>(cellVertexVelocity_vec, "cellVertexVelocity_vec.bin");
        return;
    }
#endif  

#if USE_SYCL
    sycl::buffer<vec3, 1> vertexCoord_buf(grid->vertexCoord_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size())); // CELL 顶点坐标
    sycl::buffer<vec3, 1> cellCoord_buf(grid->cellCoord_vec.data(), sycl::range<1>(grid->cellCoord_vec.size()));       // CELL 中心坐标

    sycl::buffer<size_t, 1> numberVertexOnCell_buf(grid->numberVertexOnCell_vec.data(), sycl::range<1>(grid->numberVertexOnCell_vec.size())); // CELL 有几个顶点
    sycl::buffer<size_t, 1> verticesOnCell_buf(grid->verticesOnCell_vec.data(), sycl::range<1>(grid->verticesOnCell_vec.size()));             // 
    sycl::buffer<size_t, 1> cellsOnVertex_buf(grid->cellsOnVertex_vec.data(), sycl::range<1>(grid->cellsOnVertex_vec.size()));

    sycl::buffer<vec3, 1> cellCenterVelocity_buf(cellCenterVelocity_vec.data(), sycl::range<1>(cellCenterVelocity_vec.size()));                           //CELL 中心ZTOP
    sycl::buffer<vec3, 1> cellVertexVelocity_buf(cellVertexVelocity_vec.data(), sycl::range<1>(grid->vertexCoord_vec.size() * mTotalZTopLayer));        //CELL 顶点ZTOP （要求的）


    q.submit([&](sycl::handler& cgh) {

        auto acc_vertexCoord_buf = vertexCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCoord_buf = cellCoord_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_numberVertexOnCell_buf = numberVertexOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_verticesOnCell_buf = verticesOnCell_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellsOnVertex_buf = cellsOnVertex_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellCenterVelocity_buf = cellCenterVelocity_buf.get_access<sycl::access::mode::read>(cgh);
        auto acc_cellVertexVelocity_buf = cellVertexVelocity_buf.get_access<sycl::access::mode::read_write>(cgh);


        cgh.parallel_for(sycl::range<2>(mCellsSize, mTotalZTopLayer), [=](sycl::id<2> idx) {
            size_t j = idx[0];
            size_t i = idx[1];

            auto cell_id = j;
            auto current_layer = i;

            const int CELL_SIZE = 8521;
            const int VERTEX_NUM = 6;
            const int NEIGHBOR_NUM = 3;
            const int TOTAY_ZTOP_LAYER = 60;
            const int VERTLEVELS = 60;
            // 1. 找出这个cell 的所有顶点 不存在的置为nan

            // 1.1 计算这个CELL 有多少个顶点
            auto current_cell_vertices_number = acc_numberVertexOnCell_buf[cell_id];
            // 1.2 找出所有候选顶点
            size_t current_cell_vertices_idx[VERTEX_NUM];
            for (size_t k = 0; k < VERTEX_NUM; ++k)
            {
                current_cell_vertices_idx[k] = acc_verticesOnCell_buf[cell_id * VERTEX_NUM + k] - 1; // Assuming 7 is the max number of vertices per cell
            }
            // 1.3 不存在的顶点设置为nan
            auto nan = std::numeric_limits<size_t>::max();
            for (size_t k = current_cell_vertices_number; k < VERTEX_NUM; ++k)
            {
                current_cell_vertices_idx[k] = nan;
            }
            // =============================== 找到7个顶点

            vec3 current_cell_vertices_value[VERTEX_NUM];
            bool bBoundary = false;
            for (auto k = 0; k < VERTEX_NUM; ++k)
            {
                auto vertex_idx = current_cell_vertices_idx[k];
                // 2.1 如果是nan 跳过
                if (vertex_idx == nan)
                {
                    auto double_nan = std::numeric_limits<double>::quiet_NaN();
                    current_cell_vertices_value[k] = { double_nan , double_nan , double_nan };
                    continue; 
                }
                auto current_vertex = acc_vertexCoord_buf[vertex_idx];
                // 2.2 如果不是nan 找到含有这个顶点的3个cell id(候选) 边界情况没有3个
                size_t tmp_cell_id[3];
                tmp_cell_id[0] = acc_cellsOnVertex_buf[3 * vertex_idx + 0] - 1;
                tmp_cell_id[1] = acc_cellsOnVertex_buf[3 * vertex_idx + 1] - 1;
                tmp_cell_id[2] = acc_cellsOnVertex_buf[3 * vertex_idx + 2] - 1;
                // 2.3 找到这3个CELL 的中心Velocity
                vec3 tmp_cell_center_vels[3];
                for (auto tmp_cell = 0; tmp_cell < NEIGHBOR_NUM; tmp_cell++)
                {
                    vec3 value;
                    if (tmp_cell_id[tmp_cell] > CELL_SIZE + 1)
                    {
                        value = { 0.0, 0.0, 0.0 };
                        tmp_cell_center_vels[tmp_cell] = value;
                        bBoundary = true;
                    }
                    else
                    {
                        vec3 vel;
                        auto vel_idx = VERTLEVELS * tmp_cell_id[tmp_cell] + current_layer;
                        vel = acc_cellCenterVelocity_buf[vel_idx];
                        tmp_cell_center_vels[tmp_cell] = vel;
                    }
                }
               
                // 2.4 如果是边界点
                if (bBoundary)
                {
                    current_cell_vertices_value[k] = 0.0 * tmp_cell_center_vels[0] + 0.0 * tmp_cell_center_vels[1] + 0.0 * tmp_cell_center_vels[2];
                }
                else
                {
                    double u, v, w;
                    vec3 p1 = acc_cellCoord_buf[tmp_cell_id[0]];
                    vec3 p2 = acc_cellCoord_buf[tmp_cell_id[1]];
                    vec3 p3 = acc_cellCoord_buf[tmp_cell_id[2]];
                    Interpolator::TRIANGLE tri(p1, p2, p3);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);
                    current_cell_vertices_value[k] = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
                }

                acc_cellVertexVelocity_buf[vertex_idx * TOTAY_ZTOP_LAYER + current_layer] = current_cell_vertices_value[k];
            }


            });
        });
    q.wait_and_throw();
    Debug("finished the sycl part");
    auto host_accessor = cellVertexVelocity_buf.get_access<sycl::access::mode::read>();
    auto range = host_accessor.get_range();
    size_t acc_length = range.size(); // 获取缓冲区的总大小

    std::cout << "cellVertexVelocity_buf.size() = " << cellVertexVelocity_vec.size() << " " << acc_length << std::endl;
    std::cout << "mVerticesSize x nVertLevels = " << grid->mVertexSize * mVertLevels << std::endl;
    //saveDataToTextFile2(cellVertexVelocity_vec, "OUTPUT1_ztop.txt");
    writeVertexZTopToFile<vec3>(cellVertexVelocity_vec, "cellVertexVelocity_vec.bin");
    Debug("finish Calc cellVertexVelocity_vec");
#endif


#if USE_SYCL == 0
    if (!cellVertexVelocity_vec.empty())cellVertexVelocity_vec.clear();
    cellVertexVelocity_vec.resize(grid->mVertexSize * mTotalZTopLayer);

   
    for (auto j = 0; j < mCellsSize; ++j)
    {
        for (auto i = 0; i < mTotalZTopLayer; ++i)
        {
            auto cell_id = j;
            auto current_layer = i;
            // 1.1 找出这个cell 的所有顶点
            std::vector<size_t> current_cell_vertices_idx;
            // 1.2 计算这个cell 有多少个顶点
            auto current_cell_vertices_number = grid->numberVertexOnCell_vec[cell_id];
            grid->getVerticesOnCell(cell_id, grid->verticesOnCell_vec, current_cell_vertices_idx);
            // 1.3 不存在的顶点设置为nan
            auto nan = std::numeric_limits<size_t>::max();
            if (current_cell_vertices_idx.size() != current_cell_vertices_number)
            {
                auto length = current_cell_vertices_idx.size() - current_cell_vertices_number;
                current_cell_vertices_idx.resize(current_cell_vertices_number);
                current_cell_vertices_idx.insert(current_cell_vertices_idx.end(), length, nan);
            }
            //=========================== 找到7个顶点 V

            std::vector<vec3f> current_cell_vertices_value;
            if (!current_cell_vertices_value.empty()) current_cell_vertices_value.clear();
            current_cell_vertices_value.resize(current_cell_vertices_idx.size());

            // 遍历7个顶点来计算7个顶点的ZTOP 存在current_cell_vertices_value
            bool bBoundary = false;
            for (auto k = 0; k < current_cell_vertices_idx.size(); ++k)
            {
                // 如果是nan 值 就跳过
                if (current_cell_vertices_idx[k] == nan) {
                    auto dnan = std::numeric_limits<double>::quiet_NaN();
                    current_cell_vertices_value[k] = vec3f(dnan, dnan, dnan); continue;
                };

                // 找出这个顶点的坐标
                auto vertex_idx = current_cell_vertices_idx[k];
                auto current_vertex = grid->vertexCoord_vec[vertex_idx];
                // 找出这个顶点的所有CELL 预计3个
                std::vector<size_t> tmp_cell_id;
                grid->getCellsOnVertex(vertex_idx, grid->cellsOnVertex_vec, tmp_cell_id);

                std::vector<vec3f> tmp_cell_center_vels; // 用来存相邻的这个3个CELL的中心ZTOP
                if (!tmp_cell_center_vels.empty()) tmp_cell_center_vels.clear();
                tmp_cell_center_vels.resize(3);
                // 遍历这3个CELL，分别找出存在中心的CELL的ZTOP
                for (auto tmp_cell = 0; tmp_cell < 3; tmp_cell++)
                {
                    vec3f value;
                    // 这个CELL是边界CELL，不存在3个邻接CELL
                    if (tmp_cell_id[tmp_cell] > grid->mCellsSize + 1)
                    {
                        value = vec3f(0.0f);
                        tmp_cell_center_vels[tmp_cell] = value;
                        bBoundary = true;
                    }
                    // 这个CELL 存在3个相邻的CELL
                    else
                    {
                        // 获取这个CELL在第K（current_layer）层的中心的ZTOP
                        vec3f vel;
                        getCellCenterVelocity(tmp_cell_id[tmp_cell], current_layer, cellCenterVelocity_vec, vel);
                        tmp_cell_center_vels[tmp_cell] = vel;
                    }
                }
                // 如果不是边界点
                if (!bBoundary)
                {
                    float u, v, w;
                    Interpolator::TRIANGLE tri(grid->cellCoord_vec[tmp_cell_id[0]], grid->cellCoord_vec[tmp_cell_id[1]], grid->cellCoord_vec[tmp_cell_id[2]]);
                    Interpolator::calcTriangleBarycentric(current_vertex, &tri, u, v, w);

                    current_cell_vertices_value[k] = u * tmp_cell_center_vels[0] + v * tmp_cell_center_vels[1] + w * tmp_cell_center_vels[2];
                }
                else
                {
                    // 边界点
                    current_cell_vertices_value[k] = 0.0f * tmp_cell_center_vels[0] + 0.0f * tmp_cell_center_vels[1] + 0.0f * tmp_cell_center_vels[2];
                }

                cellVertexVelocity_vec[vertex_idx * mTotalZTopLayer + current_layer] = current_cell_vertices_value[k];

            }
        }
    }

    std::cout << "cellVertexVelocity_vec.size() = " << cellVertexVelocity_vec.size() << " " << grid->vertexCoord_vec.size() * mTotalZTopLayer << std::endl;
    writeVertexZTopToFile<vec3f>(cellVertexVelocity_vec, "vertexVelocity_vec.bin");
#endif
}


