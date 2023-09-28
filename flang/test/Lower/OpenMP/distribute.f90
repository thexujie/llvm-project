! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPdistribute_simple
subroutine distribute_simple()
  ! CHECK: omp.teams
  !$omp teams

  ! CHECK: omp.distribute
  !$omp distribute

  ! CHECK: omp.wsloop
  do i = 1, 10
    call foo()
    ! CHECK: omp.yield
  end do

  ! CHECK: omp.terminator
  !$omp end distribute

  ! CHECK: omp.terminator
  !$omp end teams
end subroutine distribute_simple

!===============================================================================
! `dist_schedule` clause
!===============================================================================

! CHECK-LABEL: func @_QPdistribute_dist_schedule
! CHECK-SAME: %[[X_ARG:.*]]: !fir.ref<i32>
subroutine distribute_dist_schedule(x)
  ! CHECK: %[[X_REF:.*]]:2 = hlfir.declare %[[X_ARG]]
  integer, intent(in) :: x

  ! CHECK: omp.teams
  !$omp teams

  ! STATIC SCHEDULE, CONSTANT CHUNK SIZE

  ! CHECK: %[[CONST_CHUNK_SIZE:.*]] = arith.constant 5 : i32
  ! CHECK: omp.distribute
  ! CHECK-SAME: dist_schedule_static
  ! CHECK-SAME: chunk_size(%[[CONST_CHUNK_SIZE]] : i32)
  !$omp distribute dist_schedule(static, 5)

  ! CHECK: omp.wsloop
  do i = 1, 10
    call foo()
    ! CHECK: omp.yield
  end do

  ! CHECK: omp.terminator
  !$omp end distribute

  ! STATIC SCHEDULE, VARIABLE CHUNK SIZE

  ! CHECK: %[[X:.*]] = fir.load %[[X_REF]]#0
  ! CHECK: omp.distribute
  ! CHECK-SAME: dist_schedule_static
  ! CHECK-SAME: chunk_size(%[[X]] : i32)
  !$omp distribute dist_schedule(static, x)

  ! CHECK: omp.wsloop
  do i = 1, 10
    call foo()
    ! CHECK: omp.yield
  end do

  ! CHECK: omp.terminator
  !$omp end distribute

  ! STATIC SCHEDULE, NO CHUNK SIZE

  ! CHECK: omp.distribute
  ! CHECK-SAME: dist_schedule_static
  ! CHECK-NOT: chunk_size
  !$omp distribute dist_schedule(static)

  ! CHECK: omp.wsloop
  do i = 1, 10
    call foo()
    ! CHECK: omp.yield
  end do

  ! CHECK: omp.terminator
  !$omp end distribute

  ! CHECK: omp.terminator
  !$omp end teams
end subroutine distribute_dist_schedule

!===============================================================================
! `allocate` clause
!===============================================================================

! CHECK-LABEL: func @_QPdistribute_allocate
subroutine distribute_allocate()
  use omp_lib
  integer :: x
  ! CHECK: omp.teams
  !$omp teams

  ! CHECK: omp.distribute
  ! CHECK-SAME: allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>)
  !$omp distribute allocate(omp_high_bw_mem_alloc: x) private(x)

  ! CHECK: omp.wsloop
  do i = 1, 10
    x = i
    ! CHECK: omp.yield 
  end do

  ! CHECK: omp.terminator
  !$omp end distribute

  ! CHECK: omp.terminator
  !$omp end teams
end subroutine distribute_allocate
