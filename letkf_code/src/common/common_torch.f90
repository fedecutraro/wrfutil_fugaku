MODULE common_torch

    
    USE ftorch ! Import library for interfacing with PyTorch
    USE common

    PRIVATE
    PUBLIC ml_init, ml_routine, ml_final

    IMPLICIT NONE

    TYPE(torch_tensor), dimension(1) :: input_tensors
    TYPE(torch_tensor), dimension(1) :: output_tensors
    TYPE(torch_model) :: torch_net

    CONTAINS

    SUBROUTINE ml_init(model_torchscript_file)

        CHARACTER(LEN=128), INTENT(IN) :: model_torchscript_file

        ! Load ML model
        CALL torch_model_load(torch_net, TRIM(model_torchscript_file))
 
    END SUBROUTINE ml_init

    SUBROUTINE ml_routine(in_data, out_data)

        ! Set up Fortran data structures
        REAL(r_size), TARGET, INTENT(IN)  :: in_data(:)
        REAL(r_size), TARGET, INTENT(OUT) :: out_data(:)
        INTEGER, DIMENSION(:), ALLOCATABLE :: tensor_layout
        INTEGER :: i, ndim

        ndim = SIZE(SHAPE(in_data))
        ALLOCATE(tensor_layout(ndim))

        DO i=1,ndim:
            tensor_layout(i) = i
        END DO

        ! Create Torch input/output tensors from the above arrays
        CALL torch_tensor_from_array(input_tensors(1), in_data, tensor_layout, torch_kCPU)
        CALL torch_tensor_from_array(output_tensors(1), out_data, tensor_layout, torch_kCPU)
 
        ! Infer
        CALL torch_model_forward(torch_net, input_tensors, output_tensors)
 
    END SUBROUTINE ml_routine


    SUBROUTINE ml_final()

        ! Cleanup
        CALL torch_delete(torch_net)
        CALL torch_delete(input_tensors)
        CALL torch_delete(output_tensors)

    END SUBROUTINE ml_final

END MODULE common_torch
