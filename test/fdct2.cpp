#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void Hadamar8(double* inputs, double* outputs)
{
    double DIVISOR = sqrt(8);
    double a = inputs[0];
    double b = inputs[1];
    double c = inputs[2];
    double d = inputs[3];
    double e = inputs[4];  
    double f = inputs[5];
    double g = inputs[6];
    double h = inputs[7];
    
    outputs[0] = (a+b+c+d+e+f+g+h)/DIVISOR;
    outputs[1] = (a-b+c-d+e-f+g-h)/DIVISOR;
    outputs[2] = (a+b-c-d+e+f-g-h)/DIVISOR;
    outputs[3] = (a-b-c+d+e-f-g+h)/DIVISOR;
    outputs[4] = (a+b+c+d-e-f-g-h)/DIVISOR;
    outputs[5] = (a-b+c-d-e+f-g+h)/DIVISOR;
    outputs[6] = (a+b-c-d-e-f+g+h)/DIVISOR;
    outputs[7] = (a-b-c+d-e+f+g-h)/DIVISOR;
}

void Transform(double array[8][8])
{
    double inputs[8];
    double outputs[8];
    for(int i=0; i<8; ++i)
    {
        inputs[0] = array[i][0]; 
        inputs[1] = array[i][1];
        inputs[2] = array[i][2];
        inputs[3] = array[i][3];
        inputs[4] = array[i][4]; 
        inputs[5] = array[i][5];
        inputs[6] = array[i][6];
        inputs[7] = array[i][7];

        Hadamar8(inputs, outputs);

        array[i][0] = outputs[0]; 
        array[i][1] = outputs[1];
        array[i][2] = outputs[2];
        array[i][3] = outputs[3];
        array[i][4] = outputs[4]; 
        array[i][5] = outputs[5];
        array[i][6] = outputs[6];
        array[i][7] = outputs[7];
    }

    for(int i=0; i<8; ++i)
    {
        inputs[0] = array[0][i]; 
        inputs[1] = array[1][i];
        inputs[2] = array[2][i];
        inputs[3] = array[3][i];
        inputs[4] = array[4][i]; 
        inputs[5] = array[5][i];
        inputs[6] = array[6][i];
        inputs[7] = array[7][i];

        Hadamar8(inputs, outputs);

        array[0][i] = outputs[0]; 
        array[1][i] = outputs[1];
        array[2][i] = outputs[2];
        array[3][i] = outputs[3];
        array[4][i] = outputs[4]; 
        array[5][i] = outputs[5];
        array[6][i] = outputs[6];
        array[7][i] = outputs[7];
    }

    for(int i =0; i < 8; ++i)
    {
        printf("\n");
        for(int j =0; j< 8;++j)
        {
            printf("%f, ", array[i][j]);
        }
    }
    printf("\n");
}

int main()
{
    double array[8][8] = {{121.000000, 193.000000, 176.000000, 191.000000, 181.000000, 200.000000, 222.000000, 193.000000}, 
                     {154.000000, 187.000000, 203.000000, 228.000000, 230.000000, 160.000000, 180.000000, 213.000000},
                     {179.000000, 177.000000, 201.000000, 201.000000, 164.000000, 175.000000, 215.000000, 212.000000},
                     {170.000000, 174.000000, 186.000000, 141.000000, 231.000000, 219.000000, 219.000000, 219.000000},
                     {219.000000, 207.000000, 201.000000, 146.000000, 230.000000, 188.000000, 192.000000, 216.000000}, 
                     {209.000000, 191.000000, 210.000000, 155.000000, 155.000000, 223.000000, 179.000000, 201.000000}, 
                     {226.000000, 189.000000, 170.000000, 188.000000, 199.000000, 159.000000, 226.000000, 240.000000}, 
                     {255.000000, 138.000000, 205.000000, 178.000000, 168.000000, 208.000000, 180.000000, 206.000000}};

    for(int i =0; i < 8; ++i)
    {
        printf("\n");
        for(int j =0; j< 8;++j)
        {
            printf("%f, ", array[i][j]);
        }
    }
    Transform(array);
    Transform(array);
    
    return EXIT_SUCCESS;
}
