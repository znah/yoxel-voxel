#include "stdafx.h"
#include "path_field.h"

const float Inf = std::numeric_limits<float>::infinity();


inline float & get(Grid2Dref & grid, int2 p)
{
    return grid[p.y][p.x];
}

inline float get(const Grid2Dref & grid, int2 p)
{
    return grid[p.y][p.x];
}


struct grid_cell
{
    grid_cell(int2 p_, float v_) : p(p_), v(v_) {}
    int2 p;
    float v;
};

inline bool operator<(const grid_cell & a, const grid_cell & b)
{
    return a.v > b.v;
}

inline float get_min_val(const Grid2Dref & distmap, int2 p1, int2 p2, int & dir)
{
    float v_min = Inf;
    dir = 0;
    if (inside(distmap, p1))
    {
        float v = get(distmap, p1);
        if (v >= 0 && v < v_min)
        {
            v_min = v;
            dir = 1;
        }
    }
    if (inside(distmap, p2))
    {
        float v = get(distmap, p2);
        if (v >= 0 && v < v_min)
        {
            v_min = v;
            dir = -1;
        }

    }
    return v_min;
}


inline float estimate_dist(const Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap, int2 p)
{
    float obst_density = get(obstmap, p);
    int x = p.x, y = p.y;
    int dir_x, dir_y;
    float vx = get_min_val(distmap, int2(x+1, y), int2(x-1, y), dir_x);
    float vy = get_min_val(distmap, int2(x, y+1), int2(x, y-1), dir_y);
    assert(vx < Inf || vy < Inf);
    float v = Inf;
    float2 path(1, 1);
    if (vx == Inf || vy == Inf)
    {
        v = std::min(vx, vy) + obst_density;
    }
    else
    {
        float D = 2.0f * obst_density * obst_density - (vx-vy)*(vx-vy); 
        assert(D >= 0.0);
        v = 0.5f * (vx + vy + sqrt(D));
        path = float2(v - vx, v - vy);
        path = glm::normalize(path);
    }
    get(distmap, p) = v;
    pathmap[y][x][0] = path.x * dir_x;
    pathmap[y][x][1] = path.y * dir_y;
    return v;
}

void calc_distmap(const Grid2Dref & obstmap, Grid2Dref & distmap, V2Grid2Dref & pathmap)
{
    const int h = obstmap.shape()[0];
    const int w = obstmap.shape()[1];
    std::priority_queue<grid_cell> q;
    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
        int2 p(x, y);
        get(distmap, p) = Inf;
        pathmap[y][x][0] = 0;
        pathmap[y][x][1] = 0;
        if (get(obstmap, p) < 0)
        {
            q.push(grid_cell(p, 0.0f));
            get(distmap, p) = 0.0f;
        }
    }
    
    while (!q.empty())
    {
        grid_cell cell = q.top();
        q.pop();
        int x = cell.p.x, y = cell.p.y;
        int2 neibs[4] = {int2(x-1, y), int2(x+1, y), int2(x, y-1), int2(x, y+1)};
        for (int i = 0; i < 4; ++i)
        {
            int2 np = neibs[i];
            if (!inside(distmap, np) || get(obstmap, np) == Inf || get(distmap, np) != Inf)
                continue;
            float v = estimate_dist(obstmap, distmap, pathmap, np);
            grid_cell new_cell(np, v);
            q.push(new_cell);
        }
    }
}
