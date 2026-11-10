! GPL-2.0-or-later. Reconstruct one complete IAO channel and export W0=S0*chi0.
program qe_iao_projector
  use kinds, only: dp
  use mp_global, only: mp_startup
  use environment, only: environment_start, environment_end
  use io_global, only: ionode, ionode_id, stdout
  use mp_images, only: intra_image_comm
  use mp_bands, only: intra_bgrp_comm, root_bgrp, nbgrp
  use mp_pools, only: npool
  use mp, only: mp_bcast, mp_sum, mp_max
  use io_files, only: prefix, tmp_dir, restart_dir
  use io_base, only: write_wfc
  use pw_restart_new, only: read_collected_wfc
  use klist, only: nks, nkstot, xk, ngk, igk_k
  use wvfct, only: npwx, nbnd, current_k
  use lsda_mod, only: nspin, isk, current_spin
  use noncollin_module, only: noncolin, lspinorb, npol
  use control_flags, only: gamma_only
  use symm_base, only: nsym
  use cell_base, only: bg, tpiba
  use gvect, only: ig_l2g, mill
  use basis, only: natomwfc
  use uspp, only: nkb, vkb
  use uspp_init, only: init_us_2
  use becmod, only: becp, calbec, allocate_bec_type, deallocate_bec_type
  use wavefunctions, only: evc
  implicit none
  character(len=256) :: outdir, coeff_dir, projector_dir
  character(len=512) :: filename
  character(len=256), external :: trimcheck
  character(len=2), parameter :: updw(2)=['up','dw']
  integer :: io, ierr, ik, npw, j, header(5), maxg, nglobal, count_g, ikfile
  logical :: needwf, exists
  real(dp) :: re, im
  complex(dp) :: norm
  complex(dp), allocatable :: f(:,:), phi(:,:), w(:,:), coeff(:,:), overlap(:,:)
  integer, allocatable :: global_g(:), map(:), lookup(:), mill_k(:,:)
  complex(dp), parameter :: one=(1._dp,0._dp), zero=(0._dp,0._dp)
  namelist /iao_projector/ outdir, prefix, coeff_dir, projector_dir
  call mp_startup()
  call environment_start('QE_IAO_PROJECTOR')
  outdir='./'; prefix='pwscf'; coeff_dir=' '; projector_dir=' '; ierr=0
  if(ionode) then
    call input_from_file()
    read(5,iao_projector,iostat=ierr)
    tmp_dir=trimcheck(outdir)
  endif
  call mp_bcast(ierr,ionode_id,intra_image_comm)
  if(ierr/=0) call errore('qe_iao_projector','Invalid namelist',1)
  call mp_bcast(tmp_dir,ionode_id,intra_image_comm)
  call mp_bcast(prefix,ionode_id,intra_image_comm)
  call mp_bcast(coeff_dir,ionode_id,intra_image_comm)
  call mp_bcast(projector_dir,ionode_id,intra_image_comm)
  if(len_trim(coeff_dir)==0.or.len_trim(projector_dir)==0) &
    call errore('qe_iao_projector','Missing coefficient or output directory',1)
  if(npool/=1.or.nbgrp/=1) call errore('qe_iao_projector','Use one pool and one band group',1)
  needwf=.true.; call read_file_new(needwf)
  if(.not.needwf) call errore('qe_iao_projector','Missing collected wavefunctions',1)
  if(noncolin.or.lspinorb.or.npol/=1.or.gamma_only.or.(nspin/=1.and.nspin/=2)) &
    call errore('qe_iao_projector','Require scalar general-k collinear wavefunctions',1)
  if(nks/=nkstot.or.nsym/=1) call errore('qe_iao_projector','Require one pool and nosym',1)
  allocate(f(npwx,natomwfc),phi(npwx,1),w(npwx,1),coeff(nbnd+natomwfc,1),overlap(nbnd,1))
  allocate(global_g(npwx),map(npwx),mill_k(3,npwx))
  do ik=1,nks
    current_k=ik; current_spin=isk(ik); npw=ngk(ik)
    write(filename,'(a,"/k",i5.5,".coeff")') trim(coeff_dir),ik
    ierr=0
    if(ionode) then
      open(newunit=io,file=trim(filename),status='old',action='read',iostat=ierr)
      if(ierr==0) then
        read(io,*,iostat=ierr) header
        if(ierr==0.and.any(header/=[ik,isk(ik),nbnd,natomwfc,nks])) ierr=1
        if(ierr==0) then
          do j=1,nbnd+natomwfc
            read(io,*,iostat=ierr) re,im
            if(ierr/=0) exit
            coeff(j,1)=cmplx(re,im,dp)
          enddo
        endif
        close(io)
      endif
    endif
    call mp_bcast(ierr,ionode_id,intra_image_comm)
    if(ierr/=0) call errore('qe_iao_projector','Coefficient/header mismatch',1)
    call mp_bcast(coeff,ionode_id,intra_image_comm)
    evc=zero; f=zero; phi=zero; w=zero
    call read_collected_wfc(restart_dir(),ik,evc)
    call atomic_wfc(ik,f)
    call zgemm('N','N',npw,1,nbnd,one,evc,npwx,coeff,nbnd+natomwfc,zero,phi,npwx)
    call zgemm('N','N',npw,1,natomwfc,one,f,npwx,coeff(nbnd+1,1),nbnd+natomwfc,one,phi,npwx)
    call init_us_2(npw,igk_k(1,ik),xk(1,ik),vkb)
    call allocate_bec_type(nkb,1,becp)
    call calbec(npw,vkb,phi,becp)
    call s_psi(npwx,npw,1,phi,w)
    call deallocate_bec_type(becp)
    norm=sum(conjg(phi(1:npw,1))*w(1:npw,1))
    call mp_sum(norm,intra_bgrp_comm)
    if(.not.(abs(norm-one)<1.e-7_dp)) call errore('qe_iao_projector','Reconstructed channel is not PAW normalized',1)
    call zgemm('C','N',nbnd,1,npw,one,evc,npwx,w,npwx,zero,overlap,nbnd)
    call mp_sum(overlap,intra_bgrp_comm)
    global_g=0; mill_k=0
    do j=1,npw
      global_g(j)=ig_l2g(igk_k(j,ik)); mill_k(:,j)=mill(:,igk_k(j,ik))
    enddo
    maxg=maxval(global_g(1:npw)); call mp_max(maxg,intra_bgrp_comm)
    nglobal=npw; call mp_sum(nglobal,intra_bgrp_comm)
    allocate(lookup(maxg)); lookup=0
    do j=1,npw
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
    if(count_g/=nglobal) call errore('qe_iao_projector','Inconsistent G map',1)
    map=0; map(1:npw)=lookup(global_g(1:npw)); deallocate(lookup)
    ikfile=ik
    if(nspin==2) then
      ikfile=mod(ik-1,nks/2)+1
      write(filename,'(a,"/wfc",a,i0)') trim(projector_dir),updw(isk(ik)),ikfile
    else
      write(filename,'(a,"/wfc",i0)') trim(projector_dir),ikfile
    endif
    exists=.false.
    if(ionode) inquire(file=trim(filename)//'.dat',exist=exists)
    call mp_bcast(exists,ionode_id,intra_image_comm)
    if(exists) call errore('qe_iao_projector','Refusing to overwrite a projector file',1)
    call write_wfc(91,filename,root_bgrp,intra_bgrp_comm,ikfile,tpiba*xk(:,ik),isk(ik),nspin, &
      w,maxg,.false.,1,map,npw,tpiba*bg(:,1),tpiba*bg(:,2),tpiba*bg(:,3),mill_k,1._dp)
    if(ionode) then
      write(stdout,'(a,2i5,es25.16)') 'PROJECTOR_NORM k spin ',ik,isk(ik),real(norm,dp)
      write(filename,'(a,"/k",i5.5,".overlap")') trim(projector_dir),ik
      open(newunit=io,file=trim(filename),status='new',action='write')
      do j=1,nbnd
        write(io,'(2es25.16)') real(overlap(j,1),dp),aimag(overlap(j,1))
      enddo
      close(io)
    endif
  enddo
  call environment_end('QE_IAO_PROJECTOR')
  call stop_pp
end program
