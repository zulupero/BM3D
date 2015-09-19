/* DCT and IDCT - listing 3
 * Copyright (c) 2001 Emil Mikulic.
 * http://unix4lyfe.org/dct/
 *
 * Feel free to do whatever you like with this code.
 * Feel free to credit me.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#ifndef PI
 #ifdef M_PI
  #define PI M_PI
 #else
  #define PI 3.14159265358979
 #endif
#endif



/* Fast DCT algorithm due to Arai, Agui, Nakajima 
 * Implementation due to Tim Kientzle
 */
void dct(double array[8][8], double data[8][8])
{
	int i;
	int rows[8][8];

	static const int	c1=1004 /* cos(pi/16) << 10 */,
				s1=200 /* sin(pi/16) */,
				c3=851 /* cos(3pi/16) << 10 */,
				s3=569 /* sin(3pi/16) << 10 */,
				r2c6=554 /* sqrt(2)*cos(6pi/16) << 10 */,
				r2s6=1337 /* sqrt(2)*sin(6pi/16) << 10 */,
				r2=181; /* sqrt(2) << 7*/

	int x0,x1,x2,x3,x4,x5,x6,x7,x8;

	/* transform rows */
	for (i=0; i<8; i++)
	{
		x0 = array[0][i];
		x1 = array[1][i];
		x2 = array[2][i];
		x3 = array[3][i];
		x4 = array[4][i];
		x5 = array[5][i];
		x6 = array[6][i];
		x7 = array[7][i];

		/* Stage 1 */
		x8=x7+x0;
		x0-=x7;
		x7=x1+x6;
		x1-=x6;
		x6=x2+x5;
		x2-=x5;
		x5=x3+x4;
		x3-=x4;

		/* Stage 2 */
		x4=x8+x5;
		x8-=x5;
		x5=x7+x6;
		x7-=x6;
		x6=c1*(x1+x2);
		x2=(-s1-c1)*x2+x6;
		x1=(s1-c1)*x1+x6;
		x6=c3*(x0+x3);
		x3=(-s3-c3)*x3+x6;
		x0=(s3-c3)*x0+x6;

		/* Stage 3 */
		x6=x4+x5;
		x4-=x5;
		x5=r2c6*(x7+x8);
		x7=(-r2s6-r2c6)*x7+x5;
		x8=(r2s6-r2c6)*x8+x5;
		x5=x0+x2;
		x0-=x2;
		x2=x3+x1;
		x3-=x1;

		/* Stage 4 and output */
		rows[i][0]=x6;
		rows[i][4]=x4;
		rows[i][2]=x8>>10;
		rows[i][6]=x7>>10;
		rows[i][7]=(x2-x5)>>10;
		rows[i][1]=(x2+x5)>>10;
		rows[i][3]=(x3*r2)>>17;
		rows[i][5]=(x0*r2)>>17;
	}

	/* transform columns */
	for (i=0; i<8; i++)
	{
		x0 = rows[0][i];
		x1 = rows[1][i];
		x2 = rows[2][i];
		x3 = rows[3][i];
		x4 = rows[4][i];
		x5 = rows[5][i];
		x6 = rows[6][i];
		x7 = rows[7][i];

		/* Stage 1 */
		x8=x7+x0;
		x0-=x7;
		x7=x1+x6;
		x1-=x6;
		x6=x2+x5;
		x2-=x5;
		x5=x3+x4;
		x3-=x4;

		/* Stage 2 */
		x4=x8+x5;
		x8-=x5;
		x5=x7+x6;
		x7-=x6;
		x6=c1*(x1+x2);
		x2=(-s1-c1)*x2+x6;
		x1=(s1-c1)*x1+x6;
		x6=c3*(x0+x3);
		x3=(-s3-c3)*x3+x6;
		x0=(s3-c3)*x0+x6;

		/* Stage 3 */
		x6=x4+x5;
		x4-=x5;
		x5=r2c6*(x7+x8);
		x7=(-r2s6-r2c6)*x7+x5;
		x8=(r2s6-r2c6)*x8+x5;
		x5=x0+x2;
		x0-=x2;
		x2=x3+x1;
		x3-=x1;

		/* Stage 4 and output */
		data[0][i]=(double)((x6+16)>>3);
		data[4][i]=(double)((x4+16)>>3);
		data[2][i]=(double)((x8+16384)>>13);
		data[6][i]=(double)((x7+16384)>>13);
		data[7][i]=(double)((x2-x5+16384)>>13);
		data[1][i]=(double)((x2+x5+16384)>>13);
		data[3][i]=(double)(((x3>>8)*r2+8192)>>12);
		data[5][i]=(double)(((x0>>8)*r2+8192)>>12);
	}
}

#define COEFFS(Cu,Cv,u,v) { \
	if (u == 0) Cu = 1.0 / sqrt(2.0); else Cu = 1.0; \
	if (v == 0) Cv = 1.0 / sqrt(2.0); else Cv = 1.0; \
	}

void idct(double array[8][8], double data[8][8])
{
	int u,v,x,y;

	/* iDCT */
	for (y=0; y<8; y++)
	for (x=0; x<8; x++)
	{
		double z = 0.0;

		for (v=0; v<8; v++)
		for (u=0; u<8; u++)
		{
			double S, q;
			double Cu, Cv;
			
			COEFFS(Cu,Cv,u,v);
			S = array[v][u];

			q = Cu * Cv * S *
				cos((double)((2*x)+1) * (double)u * PI/16.0) *
				cos((double)((2*y)+1) * (double)v * PI/16.0);

			z += q;
		}

		z /= 4.0;
		if (z > 255.0) z = 255.0;
		if (z < 0) z = 0.0;

		data[y][x] = roundf(z);
	}
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
    
    double data[8][8];
    

    for(int i=0;i<8; ++i)
    {
        for(int j = 0;j<8;++j)
            printf("%f, ", array[i][j]);
        
        printf("\n");
    } 
    printf("\n\n");   
    dct(array, data);
    //dct(data, data);
    idct(data, array);
    
    /*for(int y=0; y < 8; ++y)
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

                    sum += data[v][u] * cu * cv * cos(v * ((2.0 * y) + 1.0) * M_PI / 16.0) * cos( u * ((2.0 * x) + 1.0) * M_PI / 16.0); 
                }
            }

            
            array[x][y] = fabs(roundf(0.25 * sum));
        }
    }*/

    for(int i=0;i<8; ++i)
    {
        for(int j = 0;j<8;++j)
            printf("%f, ", array[i][j]);
        
        printf("\n");
    }
	return EXIT_SUCCESS;
}