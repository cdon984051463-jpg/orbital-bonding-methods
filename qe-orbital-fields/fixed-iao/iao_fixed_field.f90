! GPL-2.0-or-later. QE 7.5: one frozen reference IAO projector per k.
! W0=S0*chi0 is stored in the native collected-WFC format. The generalized
! Hamiltonian gets lambda*W0*W0^H. W0 never follows ions or the SCF density.
module iao_fixed_field
  use kinds, only: dp
  implicit none
  private
  public :: iao_initialize, iao_hpsi, iao_observables, iao_raw_force
  logical :: active=.false.
  real(dp) :: lambda_eV=0._dp, lambda_Ry=0._dp
  character(len=256) :: projector_dir=' '
  type projector_t
    complex(dp), allocatable :: w(:,:)
  end type
  type(projector_t), allocatable :: projectors(:)
contains
  subroutine iao_initialize()
    use io_global, only: ionode, ionode_id, stdout
    use mp_images, only: intra_image_comm, nimage
    use mp_bands, only: nbgrp
    use mp_pools, only: npool
    use mp, only: mp_bcast
    use control_flags, only: restart, gamma_only, lmd, lbfgs, lscf, tprnfor, tstress
    use lsda_mod, only: nspin
    use noncollin_module, only: noncolin
    use cellmd, only: lmovecell
    use ldaU, only: lda_plus_u
    use xc_lib, only: xclib_dft_is
    use symm_base, only: nsym
    use constants, only: rytoev
    integer :: io, ierr
    namelist /iao_field/ lambda_eV, projector_dir
    ! This dedicated binary does not accept restarts, including a missing field input.
    if(restart) call errore('iao_fixed_field','Only from_scratch SCF is supported',1)
    ierr=0
    if(ionode) then
      inquire(file='iao_field.in',exist=active)
      if(active) then
        open(newunit=io,file='iao_field.in',status='old',action='read')
        read(io,iao_field,iostat=ierr)
        close(io)
      endif
    endif
    call mp_bcast(active,ionode_id,intra_image_comm)
    call mp_bcast(ierr,ionode_id,intra_image_comm)
    if(ierr/=0) call errore('iao_fixed_field','Invalid iao_field.in',1)
    if(.not.active) return
    call mp_bcast(lambda_eV,ionode_id,intra_image_comm)
    call mp_bcast(projector_dir,ionode_id,intra_image_comm)
    if(len_trim(projector_dir)==0) call errore('iao_fixed_field','Missing projector_dir',1)
    if(.not.(abs(lambda_eV)<=3._dp)) call errore('iao_fixed_field','Require finite |lambda_eV| <= 3',1)
    if((nspin/=1.and.nspin/=2).or.noncolin.or.gamma_only) &
      call errore('iao_fixed_field','Require collinear nspin=1/2 and general k',1)
    if(npool/=1.or.nbgrp/=1.or.nimage/=1) &
      call errore('iao_fixed_field','Use one pool, band group and image',1)
    if(.not.lscf.or.lmd.or.lbfgs.or.lmovecell.or..not.tprnfor.or.tstress) &
      call errore('iao_fixed_field','Only fixed-cell SCF with forces and without stress',1)
    if(nsym/=1.or.lda_plus_u.or.xclib_dft_is('hybrid').or.xclib_dft_is('meta')) &
      call errore('iao_fixed_field','Require nosym and semilocal DFT without Hubbard',1)
    lambda_Ry=lambda_eV/rytoev
    if(ionode) then
      write(stdout,'(a,es24.15)') 'IAO_FROZEN_LAMBDA_EV ',lambda_eV
      write(stdout,'(a,a)') 'IAO_FROZEN_PROJECTOR_DIR ',trim(projector_dir)
      write(stdout,'(a)') 'IAO_FROZEN_PSEUDO_OPERATOR: not an ion-following or fixed all-electron potential'
    endif
  end subroutine

  subroutine load_projector(ik,lda,n)
    use io_base, only: read_wfc
    use mp, only: mp_max, mp_sum, mp_bcast
    use mp_bands, only: root_bgrp, intra_bgrp_comm
    use klist, only: nks, xk, igk_k
    use gvect, only: ig_l2g, mill
    use cell_base, only: tpiba, bg
    use lsda_mod, only: nspin, isk
    integer, intent(in) :: ik,lda,n
    integer :: j, nglobal, maxg, expected_maxg, nb, sp, npol, ierr, count_g, ikfile
    integer, allocatable :: global_g(:), map(:), read_mill(:,:), lookup(:)
    real(dp) :: kread(3), b1(3), b2(3), b3(3), scale, error
    logical :: half
    character(len=512) :: filename
    character(len=2), parameter :: updw(2)=['up','dw']
    if(.not.allocated(projectors)) allocate(projectors(nks))
    if(allocated(projectors(ik)%w)) return
    allocate(global_g(lda),map(lda),read_mill(3,lda),projectors(ik)%w(lda,1))
    global_g=0
    do j=1,n
      global_g(j)=ig_l2g(igk_k(j,ik))
    enddo
    maxg=maxval(global_g(1:n)); call mp_max(maxg,intra_bgrp_comm)
    nglobal=n; call mp_sum(nglobal,intra_bgrp_comm)
    ! Same compact ascending-G map as QE's private gk_l2gmap_kdip.
    allocate(lookup(maxg)); lookup=0
    do j=1,n
      lookup(global_g(j))=global_g(j)
    enddo
    call mp_sum(lookup,intra_bgrp_comm)
    count_g=0
    do j=1,maxg
      if(lookup(j)==j) then
        count_g=count_g+1; lookup(j)=count_g
      else
        lookup(j)=0
      endif
    enddo
    if(count_g/=nglobal) call errore('iao_fixed_field','Inconsistent distributed G map',1)
    map=0; map(1:n)=lookup(global_g(1:n))
    ikfile=ik
    if(nspin==2) then
      ikfile=mod(ik-1,nks/2)+1
      write(filename,'(a,"/wfc",a,i0)') trim(projector_dir),updw(isk(ik)),ikfile
    else
      write(filename,'(a,"/wfc",i0)') trim(projector_dir),ikfile
    endif
    projectors(ik)%w=(0._dp,0._dp)
    nb=1; sp=isk(ik); npol=1; half=.false.; expected_maxg=maxg
    call read_wfc(91,filename,root_bgrp,intra_bgrp_comm,ikfile,kread,sp,npol, &
      projectors(ik)%w,maxg,half,nb,map,n,b1,b2,b3,read_mill,scale,ierr)
    if(ierr/=0) call errore('iao_fixed_field','Cannot read the frozen projector',1)
    call mp_bcast(b1,root_bgrp,intra_bgrp_comm)
    call mp_bcast(b2,root_bgrp,intra_bgrp_comm)
    call mp_bcast(b3,root_bgrp,intra_bgrp_comm)
    if(nb/=1.or.sp/=isk(ik).or.npol/=1.or.half.or.maxg/=expected_maxg.or.abs(scale-1._dp)>1.e-12_dp) &
      call errore('iao_fixed_field','Invalid rank-one scalar projector header',1)
    error=max(maxval(abs(kread-tpiba*xk(:,ik))),maxval(abs(b1-tpiba*bg(:,1))), &
      maxval(abs(b2-tpiba*bg(:,2))),maxval(abs(b3-tpiba*bg(:,3))))
    do j=1,n
      error=max(error,real(maxval(abs(read_mill(:,j)-mill(:,igk_k(j,ik)))),dp))
    enddo
    call mp_max(error,intra_bgrp_comm)
    if(error>1.e-9_dp) call errore('iao_fixed_field','Projector k/G/cell differs from this run',1)
    deallocate(global_g,map,read_mill,lookup)
  end subroutine

  subroutine iao_hpsi(lda,n,m,psi,hpsi)
    use wvfct, only: current_k
    use mp, only: mp_sum
    use mp_bands, only: intra_bgrp_comm
    integer, intent(in) :: lda,n,m
    complex(dp), intent(in) :: psi(lda,m)
    complex(dp), intent(inout) :: hpsi(lda,m)
    complex(dp) :: c(1,m)
    if(.not.active.or.lambda_Ry==0._dp) return
    call load_projector(current_k,lda,n)
    call zgemm('C','N',1,m,n,(1._dp,0._dp),projectors(current_k)%w,lda,psi,lda, &
      (0._dp,0._dp),c,1)
    call mp_sum(c,intra_bgrp_comm)
    call zgemm('N','N',n,m,1,cmplx(lambda_Ry,0._dp,dp),projectors(current_k)%w,lda,c,1, &
      (1._dp,0._dp),hpsi,lda)
  end subroutine

  subroutine iao_observables()
    use wvfct, only: npwx, nbnd, wg, current_k
    use klist, only: nks, ngk
    use io_files, only: nwordwfc, iunwfc
    use wavefunctions, only: evc
    use buffers, only: get_buffer
    use mp, only: mp_sum
    use mp_bands, only: intra_bgrp_comm
    use io_global, only: ionode, stdout
    use lsda_mod, only: isk
    integer :: ik,n,oldk
    complex(dp), allocatable :: c(:,:)
    real(dp) :: occupation, spin_occupation(2)
    if(.not.active) return
    allocate(c(1,nbnd)); spin_occupation=0._dp; oldk=current_k
    do ik=1,nks
      n=ngk(ik)
      if(nks>1) call get_buffer(evc,nwordwfc,iunwfc,ik)
      call load_projector(ik,npwx,n)
      call zgemm('C','N',1,nbnd,n,(1._dp,0._dp),projectors(ik)%w,npwx,evc,npwx, &
        (0._dp,0._dp),c,1)
      call mp_sum(c,intra_bgrp_comm)
      spin_occupation(isk(ik))=spin_occupation(isk(ik))+sum(wg(:,ik)*abs(c(1,:))**2)
    enddo
    if(nks>1) call get_buffer(evc,nwordwfc,iunwfc,oldk)
    occupation=sum(spin_occupation)
    if(ionode) then
      write(stdout,'(a,es25.16)') 'IAO_DEDV_ELECTRONS ',occupation
      write(stdout,'(a,2es25.16)') 'IAO_N_PER_SPIN ',spin_occupation
      write(stdout,'(a,es25.16)') 'IAO_EEXT_EV_ALREADY_IN_EBAND ',lambda_eV*occupation
      write(stdout,'(a)') 'IAO_EXPLICIT_FORCE_ZERO: frozen W0; native PAW overlap force retained'
    endif
    deallocate(c)
  end subroutine

  subroutine iao_raw_force(ipol,f)
    use io_global, only: ionode, stdout
    integer, intent(in) :: ipol
    real(dp), intent(in) :: f(:)
    if(active.and.ionode) write(stdout,'(a,i2,*(es25.16))') 'IAO_RAW_FORCE_RY_BOHR ',ipol,f
  end subroutine
end module
