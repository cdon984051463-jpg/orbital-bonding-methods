! QE 7.5 native PAW Gram exporter for the IAO density-SVD add-on.
! Uses the existing QE GPL-2.0-or-later libraries; no DFT/SCF or H reconstruction.
program qe_iao_export
  use iso_fortran_env, only: int32
  use kinds, only: dp
  use mp_global, only: mp_startup
  use environment, only: environment_start, environment_end
  use io_global, only: ionode, ionode_id, stdout
  use mp_images, only: intra_image_comm
  use mp_bands, only: intra_bgrp_comm, nbgrp
  use mp_pools, only: npool
  use mp, only: mp_bcast, mp_sum
  use io_files, only: prefix, tmp_dir, restart_dir
  use pw_restart_new, only: read_collected_wfc
  use klist, only: nks, nkstot, xk, wk, ngk, igk_k, nelec
  use wvfct, only: npwx, nbnd, et, wg, current_k
  use lsda_mod, only: nspin, isk, current_spin
  use noncollin_module, only: noncolin, lspinorb, npol
  use control_flags, only: gamma_only
  use symm_base, only: nsym
  use ions_base, only: nat, ityp, atm, tau
  use cell_base, only: at, alat
  use basis, only: natomwfc
  use uspp, only: nkb, vkb
  use uspp_param, only: upf
  use uspp_init, only: init_us_2
  use becmod, only: becp, calbec, allocate_bec_type, deallocate_bec_type
  use wavefunctions, only: evc
  implicit none
  character(len=256) :: outdir, export_dir, fname
  character(len=256), external :: trimcheck
  integer :: ios, ik, npw, na, nt, chi, l, m, col, u, npw_global, j
  integer(int32) :: head(8)
  logical :: needwf
  complex(dp), allocatable :: f(:,:), sf(:,:), sc(:,:), gcc(:,:), gcf(:,:), gff(:,:)
  complex(dp), parameter :: one=(1.0_dp,0.0_dp), zero=(0.0_dp,0.0_dp)
  real(dp) :: kcrys(3)
  namelist /iao_export/ outdir, prefix, export_dir
  call mp_startup()
  call environment_start('QE_IAO_EXPORT')
  outdir='./'; prefix='pwscf'; export_dir='./export'
  ios=0
  if (ionode) then
    call input_from_file()
    read(5,iao_export,iostat=ios)
    tmp_dir=trimcheck(outdir)
  endif
  call mp_bcast(ios,ionode_id,intra_image_comm)
  if(ios/=0) call errore('qe_iao_export','Bad namelist',1)
  call mp_bcast(tmp_dir,ionode_id,intra_image_comm)
  call mp_bcast(prefix,ionode_id,intra_image_comm)
  call mp_bcast(export_dir,ionode_id,intra_image_comm)
  if(npool/=1.or.nbgrp/=1) call errore('qe_iao_export','Use one k pool and one band group',1)
  needwf=.true.
  call read_file_new(needwf)
  if(.not.needwf) call errore('qe_iao_export','Collected wavefunctions are missing',1)
  if(noncolin.or.lspinorb.or.npol/=1.or.gamma_only) &
    call errore('qe_iao_export','v1 supports general-k scalar collinear nspin 1/2 only',1)
  if(nspin/=1.and.nspin/=2) call errore('qe_iao_export','Unsupported nspin',1)
  if(nks/=nkstot.or.nsym/=1) call errore('qe_iao_export','Require one pool and nosym identity only',1)
  if(natomwfc<1) call errore('qe_iao_export','No atomic reference functions',1)
  if(ionode) then
    open(newunit=u,file=trim(export_dir)//'/labels.tsv',status='new',action='write')
    write(u,'(a)') 'column atom species element shell l m radial'
    col=0
    do na=1,nat
      nt=ityp(na)
      do chi=1,upf(nt)%nwfc
        if(upf(nt)%oc(chi)<0.0_dp) cycle
        l=upf(nt)%lchi(chi)
        do m=1,2*l+1
          col=col+1
          write(u,'(i0,1x,i0,1x,a,1x,a,1x,a,1x,i0,1x,i0,1x,i0)') &
            col,na,trim(atm(nt)),trim(upf(nt)%psd),trim(upf(nt)%els(chi)),l,m,chi
        enddo
      enddo
    enddo
    close(u)
    if(col/=natomwfc) call errore('qe_iao_export','Reference-label count mismatch',1)
    open(newunit=u,file=trim(export_dir)//'/geometry.txt',status='new',action='write')
    write(u,*) nat,natomwfc,nbnd,nks,nspin
    write(u,'(3es25.16)') (at(:,j)*alat,j=1,3)
    do na=1,nat
      write(u,'(i0,1x,a,3es25.16)') na,trim(atm(ityp(na))),tau(:,na)*alat
    enddo
    close(u)
  endif
  allocate(f(npwx,natomwfc),sf(npwx,natomwfc),sc(npwx,nbnd))
  allocate(gcc(nbnd,nbnd),gcf(nbnd,natomwfc),gff(natomwfc,natomwfc))
  do ik=1,nks
    current_k=ik; current_spin=isk(ik)
    npw=ngk(ik)
    evc=zero
    call read_collected_wfc(restart_dir(),ik,evc)
    call atomic_wfc(ik,f)
    call init_us_2(npw,igk_k(1,ik),xk(1,ik),vkb)
    call allocate_bec_type(nkb,natomwfc,becp)
    call calbec(npw,vkb,f,becp)
    call s_psi(npwx,npw,natomwfc,f,sf)
    call deallocate_bec_type(becp)
    call allocate_bec_type(nkb,nbnd,becp)
    call calbec(npw,vkb,evc,becp)
    call s_psi(npwx,npw,nbnd,evc,sc)
    call deallocate_bec_type(becp)
    call zgemm('C','N',nbnd,nbnd,npw,one,evc,npwx,sc,npwx,zero,gcc,nbnd)
    call zgemm('C','N',nbnd,natomwfc,npw,one,evc,npwx,sf,npwx,zero,gcf,nbnd)
    call zgemm('C','N',natomwfc,natomwfc,npw,one,f,npwx,sf,npwx,zero,gff,natomwfc)
    call mp_sum(gcc,intra_bgrp_comm)
    call mp_sum(gcf,intra_bgrp_comm)
    call mp_sum(gff,intra_bgrp_comm)
    npw_global=npw
    call mp_sum(npw_global,intra_bgrp_comm)
    kcrys=matmul(transpose(at),xk(:,ik))
    if(ionode) then
      write(fname,'(a,"/k",i5.5,".gram")') trim(export_dir),ik
      open(newunit=u,file=trim(fname),access='stream',form='unformatted',status='new',action='write')
      head=int([16909060,nbnd,natomwfc,ik,isk(ik),nspin,nks,npw_global],int32)
      write(u) 'QEIAO001',head,kcrys,wk(ik),nelec,et(:,ik),wg(:,ik),gcc,gcf,gff
      close(u)
      write(stdout,'(a,4i8)') 'IAO_GRAM_WRITTEN k spin bands references ',ik,isk(ik),nbnd,natomwfc
      flush(stdout)
    endif
  enddo
  if(ionode) then
    open(newunit=u,file=trim(export_dir)//'/EXPORT_COMPLETE',status='new',action='write')
    write(u,*) nks
    close(u)
  endif
  deallocate(f,sf,sc,gcc,gcf,gff)
  call environment_end('QE_IAO_EXPORT')
  call stop_pp
end program qe_iao_export
