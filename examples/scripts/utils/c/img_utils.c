#include<stdlib.h>
#include<stdio.h>
#include<stdint.h>
#include<omp.h>

#define PERIODICNUM(row,col,slice,rows,cols,slices) (   ( (uint32_t)     (col+cols) ) %   cols +\
                                                      ( ( (uint32_t)     (row+rows) ) %   rows ) * cols +\
                                                      ( ( (uint32_t) (slice+slices) ) % slices ) * rows*cols\
                                                    )

// variables named [ out_* ] are pointers to data that will be overwritten by the function

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void map_pore_space( uint8_t *out_v, uint8_t *out_p,
                     uint32_t *out_n_pore, uint32_t *out_n_interface,
                     uint32_t rows, uint32_t cols, uint32_t slices,
                     uint8_t *mask ){

    uint32_t n_pore = 0;
    uint32_t n_interface = 0;
    uint8_t count;
    int32_t row, col;
    uint32_t id;

    #pragma omp parallel for private(row,col,count,id) reduction(+:n_pore) reduction(+:n_interface)
    for(int32_t slice=0; slice < slices; slice++){
        for(row=0; row < rows; row++){
            for(col=0; col < cols; col++){
                
                count  = ( mask[ PERIODICNUM(row-1,col  ,slice  ,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row-1,col-1,slice  ,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row  ,col-1,slice  ,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row  ,col  ,slice  ,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row-1,col  ,slice-1,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row-1,col-1,slice-1,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row  ,col-1,slice-1,rows,cols,slices) ] > 0 );
                count += ( mask[ PERIODICNUM(row  ,col  ,slice-1,rows,cols,slices) ] > 0 );

                id = col + row*cols + slice*rows*cols;

                if (count >= 8){
                    out_v[id] = 1;
                    n_pore++;
                } else if(count > 0){
                    out_p[id] = 1;
                    n_interface++;
                }
            }
        }
    }
    *out_n_pore = n_pore;
    *out_n_interface = n_interface;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

void field_to_pore_space( double *out_x,
                          double *field,
                          uint32_t rows, uint32_t cols, uint32_t slices,
                          uint8_t *mask ){

    uint32_t count=0;
    uint32_t id;
    for(uint32_t slice=0; slice < slices; slice++){
        for(uint32_t col=0; col < cols; col++){
            for(uint32_t row=0; row < rows; row++){
                id = col + row*cols + slice*rows*cols;
                if ( mask[id] ){
                    id = row + col*rows + slice*rows*cols; // accounting that field is transposed
                    out_x[ count ] = field[ id ];
                    count++;
                }
            }
        }
    }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

void make_fibers( uint8_t *out_img, 
                  uint32_t *directions, uint32_t n_fibers, float radius,
                  uint8_t fiber_color,
                  uint32_t rows, uint32_t cols, uint32_t slices ) {

    uint32_t row, col, fib;
    float dx, dy, dz, tau;

    float r2 = radius*radius;

    float *ab     = (float *)malloc(sizeof(float)*n_fibers*3);
    float *dot_ab = (float *)malloc(sizeof(float)*n_fibers);
    
    for (fib=0; fib < n_fibers; fib++){
        dx = (float) directions[ (2*fib+1)*3   ] - (float) directions[ (2*fib)*3     ];
        dy = (float) directions[ (2*fib+1)*3+1 ] - (float) directions[ (2*fib)*3 + 1 ];
        dz = (float) directions[ (2*fib+1)*3+2 ] - (float) directions[ (2*fib)*3 + 2 ];
        ab[ fib*3   ] = dx;
        ab[ fib*3+1 ] = dy;
        ab[ fib*3+2 ] = dz;
        dot_ab[ fib ] = dx*dx + dy*dy + dz*dz;
    }

    uint8_t *this_slice;

    #pragma omp parallel for private(this_slice,row,col,fib,dx,dy,dz,tau)
    for(uint32_t slice=0; slice < slices; slice++){
        this_slice = &out_img[slice*rows*cols];
        for(row=0; row < rows; row++){
            for(col=0; col < cols; col++){
                for (fib=0; fib < n_fibers; fib++){
                    dx = 0.5 + (float) col   - (float) directions[ (2*fib)*3     ];
                    dy = 0.5 + (float) row   - (float) directions[ (2*fib)*3 + 1 ];
                    dz = 0.5 + (float) slice - (float) directions[ (2*fib)*3 + 2 ];
                    tau = (dx*ab[fib*3] + dy*ab[fib*3+1] + dz*ab[fib*3+2]) / dot_ab[fib];
                    dx -= tau * ab[ fib*3   ];
                    dy -= tau * ab[ fib*3+1 ];
                    dz -= tau * ab[ fib*3+2 ];
                    tau = dx*dx + dy*dy + dz*dz; // reusing variable tau to store dist^2
                    if ( tau <= r2 ) {
                        this_slice[ col + row*cols ] = fiber_color;
                        break;
                    }
                }
            }
        }
    }
    free(ab);
    free(dot_ab);
    
}
