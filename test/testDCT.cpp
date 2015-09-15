#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    float array[8][8] = {{255.00, 0.0, 255.00, 255.00,255.00, 255.00, 255.00, 255.00},
                      {255.00, 255.00, 255.00, 255.00,0.0, 255.00, 255.00, 255.00},
                      {255.00, 255.00, 255.00, 255.00,255.00, 255.00, 255.00, 255.00},
                      {189.00, 255.00, 178.00, 255.00,255.00, 122.00, 255.00, 134.00},
                      {255.00, 255.00, 255.00, 255.00,255.00, 255.00, 255.00, 255.00},
                      {45.00, 255.00, 0.0, 255.00,255.00, 255.00, 255.00, 255.00},
                      {255.00, 45.00, 255.00, 255.00,255.00, 255.00, 255.00, 255.00},
                      {255.00, 255.00, 255.00, 255.00,0.0, 255.00, 255.00, 255.00}};
    
    float arrayOut[8][8];

    for(int v=0; v < 8; ++v)
    {
        for(int u=0; u < 8; ++u)
        {
            float sum = 0.0;
            for(int y=0; y < 8; ++y)
            {
                for(int x=0; x < 8; ++x)
                {
                    sum += array[y][x] * cos(v * ((2.0 * y) + 1.0) * M_PI / 16.0) * cos( u * ((2.0 * x) + 1.0) * M_PI / 16.0); 
                }
            }

            float cu = (u == 0) ? (1/ sqrt(2.0)) : 1.0;
            float cv = (v == 0) ? (1/ sqrt(2.0)) : 1.0;
            
            arrayOut[v][u] = 0.25 * cu * cv * sum;
        }
    }

    for(int i=0; i< 8; ++i)
    {
        for(int j=0; j< 8; ++j)
        {
            printf("%f," , arrayOut[i][j]);
        }
        printf("\n");
    }

    for(int y=0; y < 8; ++y)
    {
        for(int x=0; x < 8; ++x)
        {
            float sum = 0.0;
            for(int v=0; v < 8; ++v)
            {
                for(int u=0;u < 8; ++u)
                {
                    float cu = (u == 0) ? (1/ sqrt(2.0)) : 1.0;
                    float cv = (v == 0) ? (1/ sqrt(2.0)) : 1.0;

                    sum += arrayOut[v][u] * cu * cv * cos(v * ((2.0 * y) + 1.0) * M_PI / 16.0) * cos( u * ((2.0 * x) + 1.0) * M_PI / 16.0); 
                }
            }

            
            array[y][x] = fabs(roundf(0.25 * sum));
        }
    }

    printf("\n\nInverse\n\n");
    for(int i=0; i< 8; ++i)
    {
        for(int j=0; j< 8; ++j)
        {
            printf("%f," , array[i][j]);
        }
        printf("\n");
    }
}
