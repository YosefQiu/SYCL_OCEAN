#include "MPASOGrid.h"

MPASOGrid::MPASOGrid()
{
    mKDTree = nullptr;
}

void MPASOGrid::initGrid(MPASOReader* reader)
{
    this->mCellsSize    = reader->mCellsSize;
    this->mEdgesSize    = reader->mEdgesSize;
    this->mMaxEdgesSize = reader->mMaxEdgesSize;
    this->mVertexSize   = reader->mVertexSize;

    this->mTimesteps    = reader->mTimesteps;
    this->mVertLevels   = reader->mVertLevels;
    this->mVertLevelsP1 = reader->mVertLevelsP1;

    this->vertexCoord_vec           = std::move(reader->vertexCoord_vec);
    this->cellCoord_vec             = std::move(reader->cellCoord_vec);
    this->edgeCoord_vec             = std::move(reader->edgeCoord_vec);
    this->vertexLatLon_vec          = std::move(reader->vertexLatLon_vec);
    this->verticesOnCell_vec        = std::move(reader->verticesOnCell_vec);
    this->cellsOnVertex_vec         = std::move(reader->cellsOnVertex_vec);
    this->cellsOnCell_vec           = std::move(reader->cellsOnCell_vec);
    this->numberVertexOnCell_vec    = std::move(reader->numberVertexOnCell_vec);
    this->cellsOnEdge_vec           = std::move(reader->cellsOnEdge_vec);
    this->edgesOnCell_vec           = std::move(reader->edgesOnCell_vec);

 
}

void MPASOGrid::createKDTree(const char* kdTree_path, sycl::queue& SYCL_Q)
{
#if _WIN32 || __linux__
    std::ifstream f_in(kdTree_path, std::ifstream::binary);
    if (!f_in)
    {
        // File does not exist or cannot be read, create a new KD Tree
        mKDTree = std::make_unique<KDTree_t>(3, cellCoord_vec, 10, 5, true);
        std::ofstream f_out(kdTree_path, std::ofstream::binary);
        if (!f_out) throw std::runtime_error("Error writing index file!");
        mKDTree->index->saveIndex(f_out);
        Debug("[MPASOGrid]::Create KD Tree...");
        Debug("[MPASOGrid]::Saved KD Tree in %s", kdTree_path);
    }
    else
    {
        // File exists and can be read, load the KD Tree
        mKDTree = std::make_unique<KDTree_t>(3, cellCoord_vec, 10, 5, false);
        mKDTree->index->loadIndex(f_in);
        Debug("[MPASOGrid]::Loading KD Tree in %s", kdTree_path);
    }
#elif __APPLE__
    int n = cellCoord_vec.size();
    int neighbor_num = 1;
    int query_num = 1;
    int dim = 3;
    mKDTree = std::make_unique<kdtreegpu>(n, neighbor_num, query_num, dim, SYCL_Q, cellCoord_vec);
    mKDTree->build();
    Debug("[MPASOGrid]::Create KD Tree...");
#endif

}

void MPASOGrid::searchKDT(const CartesianCoord& point, int& cell_id)
{
#if _WIN32 || __linux__
    const int dim = 3;

    // Query point:
    std::vector<double> query_pt(dim);
    for (size_t d = 0; d < dim; d++)
        query_pt[d] = point[d];

    // do a knn search
    const size_t        num_results = 1;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<double> out_dists_sqr(num_results);

    nanoflann::KNNResultSet<double> resultSet(num_results);

    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    mKDTree->index->findNeighbors(resultSet, &query_pt[0]);

    cell_id = ret_indexes[0];
#elif __APPLE__
    float query_points[3] = { point.x(), point.y(), point.z() };
    cell_id = mKDTree->query_gpu(query_points);
#endif
}

void MPASOGrid::getNeighborCells(const size_t cell_id, std::vector<size_t>& cell_on_cell, std::vector<size_t>& neighbor_id)
{

    std::vector<size_t>::const_iterator first = cell_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!neighbor_id.empty()) neighbor_id.clear();
    neighbor_id = std::vector<size_t>(first, last);
    for (auto& val : neighbor_id) val -= 1;
}

void MPASOGrid::getVerticesOnCell(const size_t cell_id, std::vector<size_t>& vertex_on_cell, std::vector<size_t>& vertex_id)
{
    std::vector<size_t>::const_iterator first = vertex_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = vertex_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!vertex_id.empty()) vertex_id.clear();
    vertex_id = std::vector<size_t>(first, last);
    for (auto& val : vertex_id) val -= 1;
}

void MPASOGrid::getCellsOnVertex(const size_t vertex_id, std::vector<size_t>& cell_on_vertex, std::vector<size_t>& cell_id)
{
    std::vector<size_t>::const_iterator first = cell_on_vertex.begin() + (3 * vertex_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_vertex.begin() + (3 * vertex_id + 3);
    if (!cell_id.empty()) cell_id.clear();
    cell_id = std::vector<size_t>(first, last);
    for (auto& val : cell_id) val -= 1;
}

void MPASOGrid::getCellsOnEdge(const size_t edge_id, std::vector<size_t>& cell_on_edge, std::vector<size_t>& cell_id)
{
    // 输入一个边的ID，返回这个边上的两个cell的ID
    std::vector<size_t>::const_iterator first = cell_on_edge.begin() + (2 * edge_id + 0);
    std::vector<size_t>::const_iterator last = cell_on_edge.begin() + (2 * edge_id + 2);
    if (!cell_id.empty()) cell_id.clear();
    cell_id = std::vector<size_t>(first, last);
    for (auto& val : cell_id) val -= 1;
}

void MPASOGrid::getEdgesOnCell(const size_t cell_id, std::vector<size_t>& edge_on_cell, std::vector<size_t>& edge_id)
{
    // 输入一个cell的ID，返回这个cell上的所有边的ID
    std::vector<size_t>::const_iterator first = edge_on_cell.begin() + (mMaxEdgesSize * cell_id + 0);
    std::vector<size_t>::const_iterator last = edge_on_cell.begin() + (mMaxEdgesSize * cell_id + mMaxEdgesSize);
    if (!edge_id.empty()) edge_id.clear();
    edge_id = std::vector<size_t>(first, last);
    for (auto& val : edge_id) val -= 1;
} 
