! Links QE 7.5 upflib ylmr2: checks ofe_cart_from_qe_ylm (QE order/sign -> positive Cartesian)
! against the analytic harmonics, for l = 0,1,2 and 12 directions (incl. axes).  Same check that
! orbital_field_engine runs at start-up (OFE_ANGULAR_PHASE_ERROR).
program test_ofe_phase
  use ofe_math
  implicit none
  integer, parameter :: dp=ofe_dp, ng=12
  real(dp) :: g(3,ng), gg(ng), ylm(ng,9), a(5), b(5), err
  integer :: i, l, n
  g=0
  g(:,1)=[1._dp,0._dp,0._dp]; g(:,2)=[0._dp,1._dp,0._dp]; g(:,3)=[0._dp,0._dp,1._dp]
  g(:,4)=[-1._dp,0._dp,0._dp]; g(:,5)=[0._dp,-1._dp,0._dp]; g(:,6)=[0._dp,0._dp,-1._dp]
  call random_number(g(:,7:ng)); g(:,7:ng)=2*g(:,7:ng)-1
  gg=sum(g*g,dim=1)
  call ylmr2(9,ng,g,gg,ylm)
  err=0
  do i=1,ng
    do l=0,2
      n=ofe_nlm(l)
      call ofe_cart_from_qe_ylm(l,ylm(i,:),a(1:n))
      call ofe_ylm_cart(l,g(:,i),b(1:n))
      err=max(err,maxval(abs(a(1:n)-b(1:n))))
    enddo
  enddo
  if(err<1.e-12_dp) then
    write(*,'(a,es12.4)') 'PASS qe_ylmr2_to_positive_cartesian ',err
    write(*,'(a)') 'ALL_TESTS_PASS'
  else
    write(*,'(a,es12.4)') 'FAIL qe_ylmr2_to_positive_cartesian ',err
    stop 1
  endif
end program test_ofe_phase
