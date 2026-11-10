! Orbital Field Engine for QE 7.5 (model OFE_V1). GPL-2.0-or-later.
! One engine for the four orbital-field designs of the RuO2 coordination-inversion project
! (review 2026-10-25): RU_D (Ru 4d labels), O_P_BOND (O 2p sigma/pi_in/pi_out), RU_O_PI
! (joint Ru d + O p group, per-pi-channel 2x2 blocks), IAO_CHANNEL (frozen channel matrix C).
!
! A SITE is a GROUP = atoms (each with a lattice image, position tau + at*img) x shells
! (l, radial PP_CHI) with a joint S-Lowdin orthonormalisation in the PAW metric S:
!   U = S Phi G^(-1/2), G = Phi^H S Phi;  W = U C_lab;  H_ext = sum_sites W D_s W^H.
! D_s is Hermitian (complex allowed) and may differ per spin; C is shared by both spins.
! C_lab = Rblk(Q) C_loc with Q from anchors (frame moving), a fixed reference (frozen) or 1 (global).
! E_ext = sum Tr(D N) is linear in D and already in eband: it is NOT added to etot; dE/dD_ij = N_ji.
! Explicit force (fixed psi) per site and atom: phi (AO centres; zero and reported only as a
! diagnostic when CENTER frozen), aug (S moves with every PAW atom), frame (dQ/dR of C_lab),
! cder (dC_loc/dR: identically 0 in OFE_V1, C frozen).  native = raw - explicit.
! Identities printed every force call: sum_I frame_I = 0; sum_I (phi + aug)_I = -2 Re <Z,-iqU>.
! AO phase (-i)^l as QE beta (init_us_2), NOT QE atomic_wfc i^l; positive Cartesian harmonics.
! Scope: CPU, collinear nspin 1/2, general k, fixed cell, nsym = 1, no Hubbard/EXX/meta-GGA.
! V1.2 (2026-10-26): (i) OFE_HCH (print_hch): native+field KS block in the channel basis,
!   H_ch(k) = C_lab^T phit^H H phit C_lab, phit = Phi T (S-orthonormal group AOs, U = S phit), one h_psi
!   per local k in ofe_forces, k average per spin with wk/sum_spin(wk); field part in that basis = D
!   for an isolated site (U^H phit = T G T = 1).  (ii) GPU G1 host bridge (__OFE_GPU): ofe_hpsi_gpu at the
!   end of h_psi__gpu (psi device->host, dH host->device; vkb pulled only when a k cache is rebuilt).
!   print_hch = .false. leaves every V1.1 floating-point statement and record unchanged.
! V1.3 (2026-10-27): image/distance checks only on the first geometry, as OFE_WARNING (images are
!   explicit, so frames stay continuous in a relaxation); relaxation needs no extra flags
!   (allow_relax / RELAX_VALIDATED are read and ignored); cond_max is the max over all k pools.
! Input: ofe.in in the working directory (namelist &ofe_control + SITE blocks, see README).
module orbital_field_engine
  use kinds, only: dp
  use ofe_math
  implicit none
  private
  public :: ofe_initialize, ofe_hpsi, ofe_forces, ofe_record_raw_component, ofe_finish_raw, ofe_record_final
#if defined(__OFE_GPU)
  public :: ofe_hpsi_gpu
#endif
  character(len=*), parameter :: ofe_version='OFE_V1_2_20261026'
  character(len=*), parameter :: inpfile='ofe.in', statefile='ofe_state.dat'
  integer, parameter :: maxsite=16, maxatom=8, maxshell=4, maxao=48, maxch=48, maxpar=24
  integer, parameter :: maxline=3000, linelen=4096, maxsh=maxatom*maxshell   ! D_EV row of nch=48: ~2300 chars
  ! ---- namelist &ofe_control
  character(len=16) :: model='OFE_V1', control='potential'
  integer :: nsite=0
  logical :: allow_relax=.false., print_nao=.true.
  logical :: print_hch=.false., print_hch_k=.false.            ! V1.2 OFE_HCH output (shared interface spec)
  real(dp) :: dmax_eV=3._dp, v_hard_eV=10._dp, anchor_max_A=3.2_dp, group_radius_A=3.2_dp
  real(dp) :: fprint_min=1.e-6_dp
  character(len=128) :: history='ofe_history.dat'
  ! ---- sites
  type site_t
    character(len=16) :: kind=' ', frame=' ', center='moving', spin='shared'
    character(len=64) :: tag=' '
    integer :: natg=0, nao=0, nch=0, nshell=0, mode=2, ao0=0, col0=0, npar=0
    integer :: atom(maxatom)=0, img(3,maxatom)=0, nsh(maxatom)=0
    character(len=8) :: shlab(maxshell,maxatom)=' '
    character(len=4) :: elem(maxatom)='ANY', anc_elem(3)='ANY'
    integer :: shell_l(maxsh)=0, shell_start(maxsh)=0, shell_chi(maxsh)=0, shell_grp(maxsh)=0
    integer :: owner(maxao)=0
    logical :: center_frozen=.false., use_x=.true., helper_set=.false., have_fref=.false.
    logical :: have_cref(maxatom)=.false., cidentity=.false., cread=.false., relax_ok=.false.
    logical :: have_anchors=.false., dread(2)=.false.
    integer :: anc_atom(3)=0, anc_img(3,3)=0
    real(dp) :: fref(3,3)=0, cref(3,maxatom)=0, helper(3)=0
    real(dp) :: cloc(maxao,maxch)=0, clab(maxao,maxch)=0
    complex(dp) :: dch(maxch,maxch,2)=0
    character(len=32) :: pname(maxpar)=' '
    integer :: pspin(maxpar)=0
    real(dp) :: pval(maxpar)=0
    complex(dp), allocatable :: pb(:,:,:)            ! (nch,nch,maxpar), allocated at NCH
    real(dp) :: q(3,3)=0, dq(3,3,3,2)=0, condmax=0
    complex(dp), allocatable :: dao(:,:,:), ddao(:,:,:,:,:)   ! (nao,nao,2), (nao,nao,3,3,2)
  end type site_t
  type(site_t), save :: site(maxsite)
  type cache_t
    logical :: ready=.false.
    integer :: npw=-1
    complex(dp), allocatable :: phi(:,:), chi(:,:), u(:,:), w(:,:), t(:,:,:), evec(:,:,:)
    real(dp), allocatable :: eval(:,:)
  end type cache_t
  type(cache_t), allocatable, save :: cache(:)
  integer, allocatable, save :: kslot(:)
  real(dp), allocatable, save :: oldtau(:,:), raw_force(:,:), fexpl_last(:,:)
  logical, save :: first_geom=.true.
  complex(dp), allocatable, save :: dall(:,:,:)
  type(ofe_beta_t), save :: bt
  integer :: naotot=0, nchtot=0, h_apply_count=0
  integer :: h_apply_count_gpu=0                                  ! V1.2: applications through ofe_hpsi_gpu
  ! V1.2 OFE_HCH work arrays (allocated only inside ofe_forces when print_hch)
  integer, parameter :: hch_nrq_max=4                             ! occupied bands in the Rayleigh diagnostic
  complex(dp), allocatable, save :: hch_acc(:,:,:,:), hch_fac(:,:,:,:), hch_kbuf(:,:,:,:)
  integer, allocatable, save :: hch_kspin(:)
  real(dp), save :: hch_wsum(2)=0, hch_low(maxsite)=0, hch_rq=0
  integer, save :: hch_nrq=0
  logical :: initialized=.false., active=.false., nonzero=.false., any_complex=.false., raw_seen(3)=.false.
  real(dp) :: oldat(3,3)=0, oldalat=0
contains
  ! ======================================================================== small helpers
  subroutine red_c(a)
    use mp, only: mp_sum
    use mp_bands, only: intra_bgrp_comm
    complex(dp), intent(inout) :: a(:,:)
    call mp_sum(a,intra_bgrp_comm)
  end subroutine red_c

  subroutine fatal(msg,ierr)
    character(len=*), intent(in) :: msg
    integer, intent(in) :: ierr
    call errore('orbital_field_engine',msg,max(1,abs(ierr)))
  end subroutine fatal

  function upcase(s) result(r)
    character(len=*), intent(in) :: s
    character(len=len(s)) :: r
    integer :: i, c
    r=s
    do i=1,len(s)
      c=ichar(s(i:i))
      if(c>=97 .and. c<=122) r(i:i)=char(c-32)
    enddo
  end function upcase

  subroutine split_key(line,key,rest)
    character(len=*), intent(in) :: line
    character(len=*), intent(out) :: key, rest
    character(len=len(line)) :: s
    integer :: i
    s=adjustl(line)
    i=index(s,' ')
    if(i==0) i=len_trim(s)+1
    key=upcase(s(1:i-1)); rest=' '
    if(i<=len(s)) rest=adjustl(s(i:))
  end subroutine split_key

  function strip_comment(line) result(s)
    character(len=*), intent(in) :: line
    character(len=len(line)) :: s
    integer :: i1, i2, i
    s=line
    i1=index(s,'!'); i2=index(s,'#'); i=0
    if(i1>0) i=i1
    if(i2>0 .and. (i==0 .or. i2<i)) i=i2
    if(i>0) s(i:)=' '
    do i=1,len(s)
      if(ichar(s(i:i))==9) s(i:i)=' '
    enddo
  end function strip_comment

  integer function shell_l_of(lab,info)
    ! 'L0','L1','L2' (unique occupied PP_CHI of that l) or 'nS','nP','nD' (PP_CHI label)
    character(len=*), intent(in) :: lab
    integer, intent(out) :: info
    character(len=8) :: s
    integer :: n, ios
    s=upcase(adjustl(lab)); info=0; shell_l_of=-1
    if(s(1:1)=='L') then
      read(s(2:2),'(i1)',iostat=ios) n
      if(ios/=0 .or. n<0 .or. n>2) then
        info=1; return
      endif
      shell_l_of=n; return
    endif
    read(s(1:1),'(i1)',iostat=ios) n
    if(ios/=0 .or. n<1) then
      info=2; return
    endif
    select case(s(2:2))
    case('S'); shell_l_of=0
    case('P'); shell_l_of=1
    case('D'); shell_l_of=2
    case default; info=3
    end select
  end function shell_l_of

  function pos_bohr(a,img) result(p)
    use ions_base, only: tau
    use cell_base, only: at, alat
    integer, intent(in) :: a, img(3)
    real(dp) :: p(3)
    p=(tau(:,a)+matmul(at,real(img,dp)))*alat
  end function pos_bohr

  logical function elem_ok(a,e)
    use ions_base, only: ityp
    use uspp_param, only: upf
    integer, intent(in) :: a
    character(len=*), intent(in) :: e
    elem_ok=.true.
    if(upcase(trim(adjustl(e)))=='ANY') return
    elem_ok=(upcase(trim(adjustl(upf(ityp(a))%psd)))==upcase(trim(adjustl(e))))
  end function elem_ok

  ! ======================================================================== input
  subroutine read_input(nl,lines)
    use io_global, only: ionode, ionode_id
    use mp_images, only: intra_image_comm
    use mp, only: mp_bcast
    integer, intent(out) :: nl
    character(len=linelen), intent(out) :: lines(maxline)
    integer :: u, ios, i, j, flag
    integer, allocatable :: codes(:,:)
    namelist /ofe_control/ model, control, nsite, allow_relax, print_nao, dmax_eV, v_hard_eV, &
                           anchor_max_A, group_radius_A, fprint_min, history, print_hch, print_hch_k
    nl=0; ios=0; flag=0; lines=' '
    if(ionode) then
      inquire(file=inpfile,exist=active)
      if(active) then
        open(newunit=u,file=inpfile,status='old',action='read',iostat=ios)
        if(ios==0) read(u,nml=ofe_control,iostat=ios)
        if(ios==0) then
          do
            read(u,'(a)',iostat=j) lines(nl+1)
            if(j/=0) exit
            nl=nl+1
            if(nl>=maxline) then
              flag=1; exit
            endif
          enddo
          close(u)
        endif
        if(ios==0 .and. trim(model)/='OFE_V1') flag=2
        if(ios==0 .and. trim(control)/='potential') flag=3
      endif
    endif
    call mp_bcast(active,ionode_id,intra_image_comm)
    call mp_bcast(ios,ionode_id,intra_image_comm)
    call mp_bcast(flag,ionode_id,intra_image_comm)
    if(ios/=0) call fatal('Cannot read namelist &ofe_control in ofe.in',ios)
    if(flag==1) call fatal('ofe.in too long',1)
    if(flag==2) call fatal("model must be 'OFE_V1'",2)
    if(flag==3) call fatal("control must be 'potential' (occupation control = outer Python loop, "// &
                           "tools/ofe_occupation_driver.py)",3)
    if(.not.active) return
    call mp_bcast(nsite,ionode_id,intra_image_comm)
    call mp_bcast(allow_relax,ionode_id,intra_image_comm)
    call mp_bcast(print_nao,ionode_id,intra_image_comm)
    call mp_bcast(print_hch,ionode_id,intra_image_comm)
    call mp_bcast(print_hch_k,ionode_id,intra_image_comm)
    call mp_bcast(dmax_eV,ionode_id,intra_image_comm)
    call mp_bcast(v_hard_eV,ionode_id,intra_image_comm)
    call mp_bcast(anchor_max_A,ionode_id,intra_image_comm)
    call mp_bcast(group_radius_A,ionode_id,intra_image_comm)
    call mp_bcast(fprint_min,ionode_id,intra_image_comm)
    call mp_bcast(nl,ionode_id,intra_image_comm)
    if(nl>0) then
      allocate(codes(linelen,nl))
      if(ionode) then
        do j=1,nl
          do i=1,linelen
            codes(i,j)=ichar(lines(j)(i:i))
          enddo
        enddo
      endif
      call mp_bcast(codes,ionode_id,intra_image_comm)
      do j=1,nl
        do i=1,linelen
          lines(j)(i:i)=char(codes(i,j))
        enddo
      enddo
      deallocate(codes)
    endif
  end subroutine read_input

  subroutine parse_sites(nl,lines)
    integer, intent(in) :: nl
    character(len=linelen), intent(in) :: lines(maxline)
    character(len=linelen) :: ln, rest
    character(len=32) :: key, word
    character(len=8) :: labs(maxshell)
    real(dp) :: rv(2*maxch), v9(9), v3(3), re_, im_
    integer :: il, is, ios, k, g, nshl, i, ii, jj, sp, nnz, ip, info, l
    logical :: inside
    is=0; inside=.false.; il=0
    do while(il<nl)
      il=il+1
      ln=strip_comment(lines(il))
      if(len_trim(ln)==0) cycle
      call split_key(ln,key,rest)
      if(.not.inside .and. key/='SITE') call fatal('Keyword outside SITE block: '//trim(key),il)
      select case(trim(key))
      case('SITE')
        if(inside) call fatal('Nested SITE block',il)
        read(rest,*,iostat=ios) k
        is=is+1
        if(ios/=0 .or. k/=is .or. is>nsite .or. is>maxsite) call fatal('SITE index must run 1..nsite',il)
        inside=.true.
      case('END_SITE')
        inside=.false.
      case('KIND')
        read(rest,*,iostat=ios) word
        site(is)%kind=upcase(word)
      case('TAG')
        read(rest,*,iostat=ios) site(is)%tag
      case('ATOM')
        g=site(is)%natg+1
        if(g>maxatom) call fatal('Too many atoms in a group (max 8)',il)
        labs=' '
        read(rest,*,iostat=ios) site(is)%atom(g), site(is)%img(:,g), site(is)%elem(g), nshl
        if(ios==0 .and. (nshl<1 .or. nshl>maxshell)) ios=1
        if(ios==0) read(rest,*,iostat=ios) site(is)%atom(g), site(is)%img(:,g), site(is)%elem(g), nshl, &
                                          (labs(k),k=1,nshl)
        if(ios/=0) call fatal('ATOM: atom i1 i2 i3 element nshell label...',il)
        site(is)%natg=g; site(is)%nsh(g)=nshl; site(is)%shlab(1:nshl,g)=labs(1:nshl)
        do k=1,nshl
          l=shell_l_of(labs(k),info)
          if(info/=0) call fatal('Bad shell label (use L0/L1/L2 or e.g. 4D, 5S, 2P)',il)
          site(is)%nao=site(is)%nao+ofe_nlm(l)
        enddo
      case('FRAME')
        read(rest,*,iostat=ios) word
        site(is)%frame=upcase(word)
      case('ANCHORS')
        read(rest,*,iostat=ios) (site(is)%anc_atom(k), site(is)%anc_img(:,k), k=1,3)
        if(ios/=0) call fatal('ANCHORS: origin i1 i2 i3 z_from i1 i2 i3 x_from i1 i2 i3',il)
        site(is)%have_anchors=.true.
      case('ANCHOR_ELEMENTS')
        read(rest,*,iostat=ios) site(is)%anc_elem(1:3)
      case('FRAME_REF')
        read(rest,*,iostat=ios) v9
        if(ios/=0) call fatal('FRAME_REF: 9 numbers (columns x, y, z)',il)
        site(is)%fref=reshape(v9,[3,3]); site(is)%have_fref=.true.
      case('XHELPER')
        read(rest,*,iostat=ios) v3
        if(ios/=0 .or. sum(v3*v3)<1.e-12_dp) call fatal('XHELPER: 3 numbers, nonzero',il)
        site(is)%helper=v3/sqrt(sum(v3*v3)); site(is)%helper_set=.true.
      case('CENTER')
        read(rest,*,iostat=ios) word
        site(is)%center=upcase(word)
      case('CENTER_REF')
        read(rest,*,iostat=ios) g, v3
        if(ios/=0 .or. g<1 .or. g>maxatom) call fatal('CENTER_REF: g x y z (bohr)',il)
        site(is)%cref(:,g)=v3; site(is)%have_cref(g)=.true.
      case('NCH')
        read(rest,*,iostat=ios) site(is)%nch
        if(ios/=0 .or. site(is)%nch<1 .or. site(is)%nch>maxch) call fatal('NCH out of range',il)
        if(allocated(site(is)%pb)) call fatal('NCH given twice',il)
        allocate(site(is)%pb(site(is)%nch,site(is)%nch,maxpar)); site(is)%pb=(0._dp,0._dp)
      case('C_IDENTITY')
        site(is)%cidentity=.true.
      case('C_LOCAL')
        if(site(is)%nch<1 .or. site(is)%nao<1) call fatal('C_LOCAL needs ATOM lines and NCH first',il)
        if(site(is)%nao>maxao) call fatal('Too many AOs in a group',il)
        do i=1,site(is)%nao
          il=il+1
          if(il>nl) call fatal('C_LOCAL truncated',il)
          read(lines(il),*,iostat=ios) site(is)%cloc(i,1:site(is)%nch)
          if(ios/=0) call fatal('C_LOCAL row',il)
        enddo
        site(is)%cread=.true.
      case('SPIN')
        read(rest,*,iostat=ios) word
        site(is)%spin=upcase(word)
      case('D_EV')
        read(rest,*,iostat=ios) sp
        if(ios/=0 .or. sp<1 .or. sp>2) call fatal('D_EV spin (1 or 2)',il)
        if(site(is)%nch<1) call fatal('D_EV needs NCH first',il)
        do i=1,site(is)%nch
          il=il+1
          if(il>nl) call fatal('D_EV truncated',il)
          read(lines(il),*,iostat=ios) rv(1:2*site(is)%nch)
          if(ios/=0) call fatal('D_EV row: nch real parts then nch imaginary parts (eV)',il)
          site(is)%dch(i,1:site(is)%nch,sp)=cmplx(rv(1:site(is)%nch),rv(site(is)%nch+1:2*site(is)%nch),dp)
        enddo
        site(is)%dread(sp)=.true.
      case('PARAM')
        ip=site(is)%npar+1
        if(ip>maxpar) call fatal('Too many PARAM entries',il)
        if(.not.allocated(site(is)%pb)) call fatal('PARAM needs NCH first',il)
        read(rest,*,iostat=ios) site(is)%pname(ip), site(is)%pspin(ip), site(is)%pval(ip), nnz
        if(ios/=0 .or. nnz<1 .or. site(is)%pspin(ip)<0 .or. site(is)%pspin(ip)>2) &
          call fatal('PARAM name spin(0,1,2) value_eV nnz',il)
        do k=1,nnz
          il=il+1
          if(il>nl) call fatal('PARAM truncated',il)
          read(lines(il),*,iostat=ios) ii, jj, re_, im_
          if(ios/=0 .or. ii<1 .or. jj<1 .or. ii>site(is)%nch .or. jj>site(is)%nch) call fatal('PARAM entry i j re im',il)
          site(is)%pb(ii,jj,ip)=cmplx(re_,im_,dp)
        enddo
        site(is)%npar=ip
      case('RELAX_VALIDATED')
        read(rest,*,iostat=ios) site(is)%relax_ok
      case default
        call fatal('Unknown keyword in ofe.in: '//trim(key),il)
      end select
      if(ios/=0) call fatal('Cannot parse line of ofe.in',il)
    enddo
    if(inside) call fatal('Missing END_SITE',nl)
    if(is/=nsite) call fatal('Number of SITE blocks differs from nsite',is)
  end subroutine parse_sites

  ! ======================================================================== initialisation
  subroutine ofe_initialize()
    use io_global, only: ionode, ionode_id, stdout
    use ions_base, only: nat, ityp
    use uspp_param, only: upf
    use lsda_mod, only: nspin
    use klist, only: nkstot
    use control_flags, only: gamma_only, lmd, lbfgs, lscf, tstress, tprnfor, restart, isolve, scissor
    use noncollin_module, only: noncolin
    use cellmd, only: lmovecell
    use symm_base, only: nsym, time_reversal
    use mp_bands, only: nbgrp
    use mp_images, only: nimage, intra_image_comm
    use mp, only: mp_bcast
    use realus, only: real_space
    use ldaU, only: lda_plus_u
    use xc_lib, only: xclib_dft_is
    use bp, only: lelfield
    use constants, only: rytoev
    integer :: nl, is, g, k, ib, nt, l, nfound, info, ish, ao, m, sp, ip, nspin_d, col
    character(len=linelen), allocatable :: lines(:)
    character(len=8) :: s
    real(dp) :: herr, cerr, ev(maxch), rwork(3*maxch), aerr
    complex(dp) :: work(64*maxch), dtmp(maxch,maxch)
    logical :: previous_field
    if(initialized) return
    initialized=.true.
    allocate(lines(maxline))
    call read_input(nl,lines)
    if(.not.active) then
      previous_field=.false.
      if(ionode) inquire(file=statefile,exist=previous_field)
      call mp_bcast(previous_field,ionode_id,intra_image_comm)
      if(restart .and. previous_field) call fatal('field changed: ofe.in missing on restart',1)
      deallocate(lines); return
    endif
    if(nsite<1 .or. nsite>maxsite) call fatal('nsite must be 1..16',1)
    call parse_sites(nl,lines)
    deallocate(lines)
    ! ---------------------------------------------------------------- global gates
    if(noncolin .or. (nspin/=1 .and. nspin/=2)) call fatal('Only collinear nspin=1 or 2',1)
    if(gamma_only) call fatal('gamma_only is not supported (needs general k)',1)
    if(real_space) call fatal('real_space beta is not supported',1)
    if(nbgrp/=1) call fatal('Only one band group (-nb 1)',1)
    if(nimage/=1) call fatal('Only one image (no -ni / NEB)',1)
    if(lda_plus_u) call fatal('Hubbard (+U/+V) must be OFF',1)
    if(lmovecell) call fatal('Variable cell is not supported',1)
    if(lmd) call fatal('Molecular dynamics is not supported',1)
    if(xclib_dft_is('hybrid')) call fatal('Hybrid functionals / EXX are not validated',1)
    if(xclib_dft_is('meta')) call fatal('meta-GGA is not validated',1)
    if(lelfield) call fatal('lelfield (Berry-phase field) is not supported',1)
    if(lscf .and. .not.(tprnfor .or. lbfgs .or. lmd)) call fatal('Set tprnfor=.true.: N, E_ext and explicit forces are written '// &
                                           'by the force call',1)
#if (defined(__CUDA) || defined(_OPENACC)) && !defined(__OFE_GPU)
    call fatal('GPU build without __OFE_GPU: h_psi_gpu has no OFE hook (field would be skipped); '// &
               'build with the V1.2 GPU patch (apply_ofe_patch_qe75_v12.py --mode gpu)',1)
#endif
    call phase_check()
    nspin_d=nspin
    ! ---------------------------------------------------------------- per site
    naotot=0; nchtot=0
    do is=1,nsite
      associate(s_=>site(is))
      if(s_%natg<1) call fatal('SITE without ATOM lines',is)
      if(restart .or. lbfgs) then
        if(trim(s_%center)=='FROZEN' .and. .not.all(s_%have_cref(1:s_%natg))) &
          call fatal('Frozen centers need explicit CENTER_REF for restart or relaxation',is)
        if(trim(s_%frame)=='FROZEN' .and. .not.s_%have_fref) &
          call fatal('Frozen frame needs explicit FRAME_REF for restart or relaxation',is)
      endif
      select case(trim(s_%kind))
      case('RU_D','O_P_BOND','RU_O_PI','IAO_CHANNEL')
      case default
        call fatal('KIND must be RU_D, O_P_BOND, RU_O_PI or IAO_CHANNEL',is)
      end select
      select case(trim(s_%frame))
      case('GLOBAL'); s_%mode=0
      case('FROZEN'); s_%mode=1
      case('MOVING'); s_%mode=2
      case default
        call fatal('FRAME must be moving, frozen or global',is)
      end select
      select case(trim(s_%center))
      case('MOVING'); s_%center_frozen=.false.
      case('FROZEN'); s_%center_frozen=.true.
      case default
        call fatal('CENTER must be moving or frozen',is)
      end select
      select case(trim(s_%spin))
      case('SHARED','RESOLVED')
      case default
        call fatal('SPIN must be shared or resolved',is)
      end select
      if(trim(s_%spin)=='RESOLVED' .and. nspin/=2) call fatal('SPIN resolved requires nspin=2',is)
      ! group atoms, elements, shells
      ish=0; ao=0
      do g=1,s_%natg
        if(s_%atom(g)<1 .or. s_%atom(g)>nat) call fatal('Group atom index out of range',is)
        if(count(s_%atom(1:s_%natg)==s_%atom(g))/=1) call fatal('Atom appears twice in a group',is)
        if(.not.elem_ok(s_%atom(g),s_%elem(g))) &
          call fatal('Group atom element differs from UPF element (upf%psd): '//trim(s_%elem(g)),is)
        nt=ityp(s_%atom(g))
        do k=1,s_%nsh(g)
          l=shell_l_of(s_%shlab(k,g),info)
          s=upcase(adjustl(s_%shlab(k,g)))
          nfound=0
          do ib=1,upf(nt)%nwfc
            if(upf(nt)%lchi(ib)/=l .or. upf(nt)%oc(ib)<0._dp) cycle
            if(s(1:1)/='L') then
              if(upcase(trim(adjustl(upf(nt)%els(ib))))/=trim(s)) cycle
            endif
            nfound=nfound+1
            if(nfound==1) then
              ish=ish+1
              s_%shell_l(ish)=l; s_%shell_start(ish)=ao+1; s_%shell_chi(ish)=ib; s_%shell_grp(ish)=g
              do m=1,ofe_nlm(l)
                s_%owner(ao+m)=s_%atom(g)
              enddo
              ao=ao+ofe_nlm(l)
            endif
          enddo
          if(nfound/=1) call fatal('Shell must match exactly one PP_CHI with oc>=0: '//trim(s),is)
        enddo
      enddo
      s_%nshell=ish
      if(ao/=s_%nao .or. ao>maxao) call fatal('AO count mismatch / too many AOs',is)
      if(s_%cidentity) then
        if(s_%cread) call fatal('Give either C_IDENTITY or C_LOCAL',is)
        if(s_%nch==0) s_%nch=s_%nao
        if(s_%nch/=s_%nao) call fatal('C_IDENTITY requires NCH = nao',is)
        s_%cloc=0
        do k=1,s_%nao
          s_%cloc(k,k)=1
        enddo
      else if(.not.s_%cread) then
        call fatal('Channel matrix missing (C_IDENTITY or C_LOCAL)',is)
      endif
      if(s_%nch>s_%nao) call fatal('NCH > nao',is)
      if(.not.allocated(s_%pb)) then
        allocate(s_%pb(s_%nch,s_%nch,maxpar)); s_%pb=(0._dp,0._dp)
      endif
      allocate(s_%dao(s_%nao,s_%nao,2),s_%ddao(s_%nao,s_%nao,3,3,2))
      s_%dao=(0._dp,0._dp); s_%ddao=(0._dp,0._dp)
      cerr=maxval(abs(matmul(transpose(s_%cloc(1:s_%nao,1:s_%nch)),s_%cloc(1:s_%nao,1:s_%nch))-eye_r(s_%nch)))
      if(cerr>1.e-8_dp) call fatal('C_LOCAL columns are not orthonormal (1e-8)',is)
      ! kind-specific structure
      select case(trim(s_%kind))
      case('RU_D')
        if(s_%natg/=1 .or. s_%nshell/=1 .or. s_%shell_l(1)/=2) call fatal('RU_D: one atom, one l=2 shell',is)
      case('O_P_BOND')
        if(s_%natg/=1 .or. s_%nshell/=1 .or. s_%shell_l(1)/=1) call fatal('O_P_BOND: one atom, one l=1 shell',is)
        if(s_%mode==2) then
          if(s_%anc_atom(2)/=s_%atom(1) .or. any(s_%anc_img(:,2)/=s_%img(:,1))) &
            call fatal('O_P_BOND: z_from anchor must be the site O (same image); origin = axis partner',is)
        endif
      case('RU_O_PI')
        if(s_%natg/=2 .or. s_%nshell/=2 .or. s_%shell_l(1)/=2 .or. s_%shell_l(2)/=1) &
          call fatal('RU_O_PI: atom 1 one l=2 shell, atom 2 one l=1 shell',is)
        if(s_%mode/=0) then
          if(s_%anc_atom(1)/=s_%atom(1) .or. any(s_%anc_img(:,1)/=s_%img(:,1)) .or. &
             s_%anc_atom(2)/=s_%atom(2) .or. any(s_%anc_img(:,2)/=s_%img(:,2))) &
            call fatal('RU_O_PI: anchors origin/z_from must be the group Ru and O (same images)',is)
          if(s_%anc_atom(3)==0) call fatal('RU_O_PI: x_from (side O) required',is)
        endif
      end select
      ! frame anchors
      if(s_%mode==2 .or. (s_%mode==1 .and. .not.s_%have_fref)) then
        if(.not.s_%have_anchors) call fatal('ANCHORS required for frame moving (or frozen without FRAME_REF)',is)
      endif
      if(s_%have_anchors) then
        do k=1,3
          if(k==3 .and. s_%anc_atom(3)==0) cycle
          if(s_%anc_atom(k)<1 .or. s_%anc_atom(k)>nat) call fatal('Anchor atom index out of range',is)
          if(.not.elem_ok(s_%anc_atom(k),s_%anc_elem(k))) &
            call fatal('Anchor element differs from UPF element: '//trim(s_%anc_elem(k)),is)
        enddo
        if(s_%anc_atom(1)==s_%anc_atom(2) .and. all(s_%anc_img(:,1)==s_%anc_img(:,2))) &
          call fatal('origin and z_from anchors coincide',is)
        s_%use_x=(s_%anc_atom(3)/=0)
        if(s_%use_x) then
          if((s_%anc_atom(3)==s_%anc_atom(1) .and. all(s_%anc_img(:,3)==s_%anc_img(:,1))) .or. &
             (s_%anc_atom(3)==s_%anc_atom(2) .and. all(s_%anc_img(:,3)==s_%anc_img(:,2)))) &
            call fatal('x_from anchor coincides with origin or z_from',is)
        endif
      endif
      if(s_%have_fref) then
        if(maxval(abs(matmul(transpose(s_%fref),s_%fref)-eye_r(3)))>1.e-8_dp) &
          call fatal('FRAME_REF not orthonormal (1e-8)',is)
      endif
      if(s_%center_frozen) then
        do g=1,s_%natg
          if(.not.s_%have_cref(g) .and. ionode) write(stdout,'(a,2i4)') &
            'OFE_WARNING CENTER frozen without CENTER_REF: initial position used for site/atom ',is,g
        enddo
      endif
      ! D (eV -> Ry), parameters
      if(.not.s_%dread(1)) s_%dch(:,:,1)=0
      if(trim(s_%spin)=='SHARED') then
        if(s_%dread(2)) call fatal('D_EV 2 given but SPIN shared',is)
        s_%dch(:,:,2)=s_%dch(:,:,1)
      else if(.not.s_%dread(2)) then
        s_%dch(:,:,2)=0
      endif
      if(any(abs(s_%dch(s_%nch+1:,:,:))>0._dp) .or. any(abs(s_%dch(:,s_%nch+1:,:))>0._dp)) &
        call fatal('D entries beyond NCH',is)
      do sp=1,2
        herr=maxval(abs(s_%dch(1:s_%nch,1:s_%nch,sp)-transpose(conjg(s_%dch(1:s_%nch,1:s_%nch,sp)))))
        if(herr>1.e-12_dp) call fatal('D is not Hermitian',is)
        dtmp(1:s_%nch,1:s_%nch)=s_%dch(1:s_%nch,1:s_%nch,sp)
        call zheev('N','U',s_%nch,dtmp,maxch,ev,work,size(work),rwork,info)
        if(info/=0) call fatal('zheev(D) failed',info)
        if(maxval(abs(ev(1:s_%nch)))>min(dmax_eV,v_hard_eV)+1.e-10_dp) &
          call fatal('|eig D| exceeds dmax_eV (project cap 3 eV; hard cap v_hard_eV = 10 eV)',is)
        if(any(abs(aimag(s_%dch(1:s_%nch,1:s_%nch,sp)))>0._dp)) any_complex=.true.
      enddo
      do ip=1,s_%npar
        herr=maxval(abs(s_%pb(1:s_%nch,1:s_%nch,ip)-transpose(conjg(s_%pb(1:s_%nch,1:s_%nch,ip)))))
        if(herr>1.e-12_dp) call fatal('PARAM basis matrix is not Hermitian: '//trim(s_%pname(ip)),is)
      enddo
      s_%dch=s_%dch/rytoev
      if(any(abs(s_%dch(1:s_%nch,1:s_%nch,1:nspin))>0._dp)) nonzero=.true.
      ! helper x axis: field must be axially symmetric about local z
      if(s_%have_anchors .and. .not.s_%use_x .and. s_%mode/=0) then
        do sp=1,nspin
          aerr=ofe_axial_err(s_%nao,s_%nch,s_%nshell,s_%shell_l(1:s_%nshell),s_%shell_start(1:s_%nshell), &
                             s_%cloc(1:s_%nao,1:s_%nch),s_%dch(1:s_%nch,1:s_%nch,sp))
          if(aerr>1.e-10_dp) call fatal('x_from = 0 (helper axis) needs a field symmetric about z',is)
        enddo
      endif
      s_%ao0=naotot; s_%col0=nchtot
      naotot=naotot+s_%nao; nchtot=nchtot+s_%nch
      end associate
    enddo
    ! ---------------------------------------------------------------- field-dependent gates
    if(nonzero .and. nsym>1) call fatal('Nonzero field needs nsym=1 (set nosym=.true.)',nsym)
    if(nsym>1 .and. ionode) write(stdout,'(a)') 'OFE_WARNING nsym>1 with zero field: N is an irreducible-wedge sum'
    if(any_complex .and. time_reversal) call fatal('Complex D breaks time reversal: set noinv=.true.',1)
    if(nonzero .and. tstress) call fatal('Stress is not implemented: tstress=.false.',1)
    if(dmax_eV>3._dp .and. ionode) write(stdout,'(a,f8.3)') 'OFE_WARNING dmax_eV above project cap 3 eV: ',dmax_eV
    if(print_hch_k .and. .not.print_hch) call fatal('print_hch_k requires print_hch=.true.',1)
    if(print_hch) then
      if(nsym>1) call fatal('print_hch needs nsym=1 (nosym=.true.): H_ch is a k average in a site basis, '// &
                            'not symmetrised',nsym)
      if(scissor) call fatal('print_hch with the scissor operator is not supported',1)
    endif
#if defined(__OFE_GPU)
    if(nonzero .and. isolve/=0) call fatal("__OFE_GPU build with a nonzero field: diagonalization='david' "// &
                                           "required (the h_psi_gpu hook is audited for Davidson only)",isolve)
    if(ionode) write(stdout,'(a,a)') 'OFE_GPU_ROUTE BRIDGE_G1 ',ofe_version
#endif
    allocate(dall(nchtot,nchtot,2)); dall=0
    do is=1,nsite
      col=site(is)%col0
      do sp=1,2
        dall(col+1:col+site(is)%nch,col+1:col+site(is)%nch,sp)=site(is)%dch(1:site(is)%nch,1:site(is)%nch,sp)
      enddo
    enddo
    call state_check(restart)
    if(ionode) call write_header(nspin_d,nkstot,nsym,time_reversal,restart)
  contains
    function eye_r(n) result(a)
      integer, intent(in) :: n
      real(dp) :: a(n,n)
      integer :: i
      a=0
      do i=1,n
        a(i,i)=1
      enddo
    end function eye_r
  end subroutine ofe_initialize

  subroutine phase_check()
    ! QE ylmr2 -> positive Cartesian mapping must reproduce the analytic harmonics.
    use io_global, only: ionode, stdout
    real(dp) :: q(3,4), q2(4), y(4,9), a(5), b(5), err
    integer :: i, l
    q(:,1)=[1._dp,0._dp,0._dp]; q(:,2)=[0._dp,1._dp,0._dp]
    q(:,3)=[0._dp,0._dp,1._dp]; q(:,4)=[.2_dp,-.3_dp,.7_dp]
    q2=sum(q*q,dim=1)
    call ylmr2(9,4,q,q2,y)
    err=0
    do i=1,4
      do l=0,2
        call ofe_cart_from_qe_ylm(l,y(i,:),a(1:2*l+1))
        call ofe_ylm_cart(l,q(:,i),b(1:2*l+1))
        err=max(err,maxval(abs(a(1:2*l+1)-b(1:2*l+1))))
      enddo
    enddo
    if(err>1.e-10_dp) call fatal('Real-harmonic phase mapping failed',1)
    if(ionode) write(stdout,'(a,es12.3)') 'OFE_ANGULAR_PHASE_ERROR ',err
  end subroutine phase_check

  subroutine state_check(restart)
    ! Stale-state protection: ofe.in is copied to ofe_state.dat. restart_mode='restart' requires an
    ! identical copy (else stop); from_scratch replaces it (OFE_STATE_REPLACED if it differed).
    use io_global, only: ionode, ionode_id, stdout
    use mp_images, only: intra_image_comm
    use mp, only: mp_bcast
    logical, intent(in) :: restart
    character(len=linelen) :: a, b
    integer :: u1, u2, i1, i2, flag
    logical :: ex, same
    flag=0
    if(ionode) then
      inquire(file=statefile,exist=ex)
      same=ex
      if(ex) then
        open(newunit=u1,file=inpfile,status='old',action='read')
        open(newunit=u2,file=statefile,status='old',action='read')
        do
          read(u1,'(a)',iostat=i1) a
          read(u2,'(a)',iostat=i2) b
          if(i1/=0 .or. i2/=0) then
            if(i1==0 .or. i2==0) same=.false.
            exit
          endif
          if(trim(a)/=trim(b)) then
            same=.false.; exit
          endif
        enddo
        close(u1); close(u2)
      endif
      if(restart) then
        if(.not.ex) flag=1
        if(ex .and. .not.same) flag=2
      else
        if(ex .and. .not.same) write(stdout,'(a)') 'OFE_STATE_REPLACED (from_scratch with a different ofe.in)'
        call copy_file(inpfile,statefile)
      endif
    endif
    call mp_bcast(flag,ionode_id,intra_image_comm)
    if(flag==1) call fatal('Restart: ofe_state.dat missing (field state of the interrupted run unknown)',1)
    if(flag==2) call fatal('Restart: field changed since the interrupted run (ofe.in /= ofe_state.dat)',2)
  end subroutine state_check

  subroutine copy_file(src,dst)
    character(len=*), intent(in) :: src, dst
    character(len=linelen) :: a
    integer :: u1, u2, ios
    open(newunit=u1,file=src,status='old',action='read')
    open(newunit=u2,file=dst,status='replace',action='write')
    do
      read(u1,'(a)',iostat=ios) a
      if(ios/=0) exit
      write(u2,'(a)') trim(a)
    enddo
    close(u1); close(u2)
  end subroutine copy_file

  subroutine write_header(nsp_,nkt,nsy,trev,rst)
    use ions_base, only: nat
    use constants, only: rytoev
    integer, intent(in) :: nsp_, nkt, nsy
    logical, intent(in) :: trev, rst
    integer :: u, is, g, i, ip, sp
    open(newunit=u,file=trim(history),status='unknown',position='append',action='write')
    write(u,'(a,a)') 'OFE_HEADER version ',ofe_version
    write(u,'(a)') 'OFE_HEADER units energy Ry; force Ry/bohr (F = -dE/dR); position bohr; D printed in eV'
    write(u,'(a)') 'OFE_HEADER N_ch = C_lab^T N_ao C_lab, N_ao(i,j) = sum_k,n w <u_i|psi><psi|u_j>; '// &
                   'nspin=1: spin-summed (0..2), nspin=2: per spin (0..1); E_ext = Tr(D N) already in eband'
    write(u,'(a)') 'OFE_HEADER OFE_N = Re N (time-reversal-reduced k sums give spurious Im; see OFE_N_IMAG_RAW_MAX)'
    write(u,'(a)') 'OFE_HEADER OFE_FORCE_SPLIT site atom phi(3) aug(3) frame(3) cder(3): phi = AO-centre motion '// &
                   '(0 if CENTER frozen), aug = PAW S/beta motion (all PAW atoms), frame = dQ/dR, cder = 0 (C frozen)'
    write(u,'(a)') 'OFE_HEADER identities: sum_I frame_I = 0; sum_I (phi or phi_diag + aug)_I = -2 Re <Z,-iqU> '// &
                   '(OFE_SUMRULE); sum_I explicit_I is NOT zero in general'
    write(u,'(a)') 'OFE_HEADER native = raw - explicit_total; raw = QE total force before net-force removal'
    if(print_hch) then
      write(u,'(a)') 'OFE_HEADER OFE_HCH_UNITS Ry: OFE_HCH = Re sum_k wbar_k C_lab^T phit^H H phit C_lab, '// &
                     'phit = Phi T, wbar = wk/sum_spin(wk), H includes the OFE field; OFE_HCH_FIELD = D'
      write(u,'(a)') 'OFE_HCH_UNITS Ry'
    endif
#if defined(__OFE_GPU)
    write(u,'(a,a)') 'OFE_GPU_ROUTE BRIDGE_G1 ',ofe_version
#endif
    write(u,'(a,9i8,2l3)') 'OFE_RUN nspin nkstot nsym nat nsite naotot nchtot 0 0 trev restart ', &
      nsp_, nkt, nsy, nat, nsite, naotot, nchtot, 0, 0, trev, rst
    do is=1,nsite
      associate(s_=>site(is))
      write(u,'(a,i4,1x,a,1x,a,1x,a,1x,a,1x,a,3i5)') 'OFE_SITE ',is,trim(s_%kind),trim(s_%tag),trim(s_%frame), &
        trim(s_%center),trim(s_%spin),s_%natg,s_%nao,s_%nch
      do g=1,s_%natg
        write(u,'(a,2i4,i7,3i4,1x,a,1x,i3,4(1x,a))') 'OFE_SITE_ATOM ',is,g,s_%atom(g),s_%img(:,g),trim(s_%elem(g)), &
          s_%nsh(g),(trim(s_%shlab(i,g)),i=1,s_%nsh(g))
      enddo
      if(s_%have_anchors) write(u,'(a,i4,3(i7,3i4),l3)') 'OFE_SITE_ANCHORS ',is, &
        (s_%anc_atom(i),s_%anc_img(:,i),i=1,3),s_%use_x
      do sp=1,2
        do i=1,s_%nch
          write(u,'(a,3i4,*(es20.11))') 'OFE_DLOC_EV site spin row re im ',is,sp,i, &
            real(s_%dch(i,1:s_%nch,sp),dp)*rytoev,aimag(s_%dch(i,1:s_%nch,sp))*rytoev
        enddo
      enddo
      do ip=1,s_%npar
        write(u,'(a,i4,1x,a,i3,es20.11)') 'OFE_PARAM site name spin value_eV ',is,trim(s_%pname(ip)), &
          s_%pspin(ip),s_%pval(ip)
      enddo
      end associate
    enddo
    close(u)
  end subroutine write_header

  ! ======================================================================== geometry / caches
  subroutine nearest_image_check(ref,a,img,is,what)
    use io_global, only: ionode, stdout
    ! warn if another lattice image of atom a is closer to ref than the requested one
    real(dp), intent(in) :: ref(3)
    integer, intent(in) :: a, img(3), is
    character(len=*), intent(in) :: what
    real(dp) :: d0, d, p(3)
    integer :: i1, i2, i3
    d0=sqrt(sum((pos_bohr(a,img)-ref)**2))
    do i1=-1,1
      do i2=-1,1
        do i3=-1,1
          if(i1==0 .and. i2==0 .and. i3==0) cycle
          p=pos_bohr(a,img+[i1,i2,i3])
          d=sqrt(sum((p-ref)**2))
          if(d<d0-1.e-6_dp) then
            if(ionode) write(stdout,'(a,i4,1x,a)') 'OFE_WARNING not the nearest image: site',is,what
            return
          endif
        enddo
      enddo
    enddo
  end subroutine nearest_image_check

  subroutine ofe_refresh()
    use ions_base, only: nat, tau, ityp
    use cell_base, only: at, alat
    use klist, only: nks, ngk, xk, igk_k
    use lsda_mod, only: nspin
    use uspp, only: nkb, okvan, qq_at, ofsbeta
    use uspp_param, only: upf, nh, nhm
    use constants, only: bohr_radius_angs
    use io_global, only: ionode, stdout
    integer :: is, g, k, sp, info, jk, j
    real(dp) :: p0(3), d, a(3), b(3), po(3), pz(3), px(3)
    if(.not.allocated(oldtau)) then
      allocate(oldtau(3,nat),raw_force(3,nat),fexpl_last(3,nat),cache(nks),kslot(nks))
      oldtau=huge(1._dp); oldat=at; oldalat=alat; raw_force=0; fexpl_last=0
      ! LSDA: spin-up and spin-down copies of the same k share one projector cache
      do k=1,nks
        kslot(k)=k
        do jk=1,k-1
          if(kslot(jk)/=jk) cycle
          if(ngk(jk)/=ngk(k)) cycle
          if(maxval(abs(xk(:,jk)-xk(:,k)))>1.e-12_dp) cycle
          if(any(igk_k(1:ngk(k),jk)/=igk_k(1:ngk(k),k))) cycle
          kslot(k)=jk; exit
        enddo
      enddo
      bt%nat=nat; bt%nhm=nhm; bt%nkb=nkb
      allocate(bt%ofs(nat),bt%nh(nat),bt%us(nat),bt%qq(nhm,nhm,nat))
      do j=1,nat
        bt%ofs(j)=ofsbeta(j); bt%nh(j)=nh(ityp(j)); bt%us(j)=okvan .and. upf(ityp(j))%tvanp
      enddo
      bt%qq=qq_at(1:nhm,1:nhm,1:nat)
    endif
    if(size(cache)/=nks) call fatal('k list changed',1)
    if(maxval(abs(at-oldat))>1.e-12_dp .or. abs(alat-oldalat)>1.e-12_dp) call fatal('Cell changed (fixed-cell model)',1)
    if(all(tau==oldtau)) return
    do k=1,nks
      cache(k)%ready=.false.
    enddo
    do is=1,nsite
      associate(s_=>site(is))
      p0=pos_bohr(s_%atom(1),s_%img(:,1))
      do g=2,s_%natg
        d=sqrt(sum((pos_bohr(s_%atom(g),s_%img(:,g))-p0)**2))*bohr_radius_angs
        if(first_geom) then
          if(d>group_radius_A .and. ionode) write(stdout,'(a,i4,f9.4)') &
            'OFE_WARNING group atom farther than group_radius_A from atom 1: site, d/A',is,d
          call nearest_image_check(p0,s_%atom(g),s_%img(:,g),is,'group atom')
        endif
      enddo
      if(s_%center_frozen) then
        do g=1,s_%natg
          if(.not.s_%have_cref(g)) then
            s_%cref(:,g)=pos_bohr(s_%atom(g),s_%img(:,g)); s_%have_cref(g)=.true.
            if(ionode) write(stdout,'(a,2i4,3f16.10)') 'OFE_CENTER_REF_FROM_INITIAL_GEOMETRY ',is,g,s_%cref(:,g)
          endif
        enddo
      endif
      s_%q=0; s_%dq=0
      do k=1,3
        s_%q(k,k)=1
      enddo
      if(s_%have_anchors .and. (s_%mode==2 .or. .not.s_%have_fref)) then
        po=pos_bohr(s_%anc_atom(1),s_%anc_img(:,1)); pz=pos_bohr(s_%anc_atom(2),s_%anc_img(:,2))
        a=pz-po
        if(first_geom) then
          if(sqrt(sum(a*a))*bohr_radius_angs>anchor_max_A .and. ionode) write(stdout,'(a,i4)') &
            'OFE_WARNING z_from anchor farther than anchor_max_A: site',is
          call nearest_image_check(po,s_%anc_atom(2),s_%anc_img(:,2),is,'z_from anchor')
        endif
        if(s_%use_x) then
          px=pos_bohr(s_%anc_atom(3),s_%anc_img(:,3)); b=px-po
          if(first_geom) then
            if(min(sqrt(sum(b*b)),sqrt(sum((px-pz)**2)))*bohr_radius_angs>anchor_max_A .and. ionode) &
              write(stdout,'(a,i4)') 'OFE_WARNING x_from anchor farther than anchor_max_A: site',is
            if(sqrt(sum(b*b))<=sqrt(sum((px-pz)**2))) then
              call nearest_image_check(po,s_%anc_atom(3),s_%anc_img(:,3),is,'x_from anchor')
            else
              call nearest_image_check(pz,s_%anc_atom(3),s_%anc_img(:,3),is,'x_from anchor')
            endif
          endif
        else
          if(.not.s_%helper_set) then
            k=minloc(abs(a),dim=1)
            s_%helper=0; s_%helper(k)=1; s_%helper_set=.true.
          endif
          b=s_%helper
        endif
        call ofe_frame(a,b,s_%q,s_%dq,info)
        if(info/=0) call fatal('Zero z anchor vector or collinear frame anchors',info)
        if(s_%mode==1) then
          s_%fref=s_%q; s_%have_fref=.true.
          if(ionode) write(stdout,'(a,i4,9f14.10)') 'OFE_FRAME_REF_FROM_INITIAL_GEOMETRY ',is,s_%fref
        endif
      endif
      if(s_%mode==1) then
        s_%q=s_%fref; s_%dq=0
      endif
      do sp=1,nspin
        call ofe_dao(s_%nao,s_%nch,s_%nshell,s_%shell_l(1:s_%nshell),s_%shell_start(1:s_%nshell), &
                     s_%cloc(1:s_%nao,1:s_%nch),s_%dch(1:s_%nch,1:s_%nch,sp),s_%mode,s_%q,s_%dq,s_%use_x, &
                     s_%clab(1:s_%nao,1:s_%nch),s_%dao(1:s_%nao,1:s_%nao,sp),s_%ddao(1:s_%nao,1:s_%nao,:,:,sp))
      enddo
      end associate
    enddo
    oldtau=tau; first_geom=.false.
  end subroutine ofe_refresh

  subroutine ofe_prepare(ik,n,lda)
    use ions_base, only: nsp, ityp
    use uspp, only: vkb
    use uspp_param, only: nwfcm
    use atwfc_mod, only: interp_atwfc
    use gvect, only: g
    use klist, only: xk, igk_k
    use cell_base, only: tpiba
    integer, intent(in) :: ik, n, lda
    real(dp), allocatable :: q(:,:), q2(:), qlen(:), ylm(:,:), yc(:,:), chiq(:,:,:)
    complex(dp), allocatable :: sk(:), cl(:,:)
    real(dp) :: p(3), cond
    complex(dp) :: lph
    integer :: s, is, ish, i, l, m, nt, col, info, ga, a0
    call ofe_refresh()
    s=kslot(ik)
    if(cache(s)%ready) then
      if(cache(s)%npw/=n) call fatal('Number of plane waves changed for a k point',1)
      return
    endif
    if(.not.allocated(cache(s)%phi)) then
      allocate(cache(s)%phi(lda,naotot),cache(s)%chi(lda,naotot),cache(s)%u(lda,naotot),cache(s)%w(lda,nchtot))
      allocate(cache(s)%t(maxao,maxao,nsite),cache(s)%evec(maxao,maxao,nsite),cache(s)%eval(maxao,nsite))
    endif
    if(size(cache(s)%phi,1)/=lda) call fatal('Plane-wave leading dimension changed',1)
    allocate(q(3,n),q2(n),qlen(n),ylm(n,9),yc(n,9),chiq(n,nwfcm,nsp),sk(n))
    do i=1,n
      q(:,i)=tpiba*(g(:,igk_k(i,ik))+xk(:,ik))
    enddo
    q2=sum((q/tpiba)**2,dim=1); qlen=sqrt(q2)*tpiba
    call ylmr2(9,n,q/tpiba,q2,ylm)
    ! QE (+z,-x,-y | +z2,-xz,-yz,+x2-y2,+xy) -> positive Cartesian (x,y,z | z2,xz,yz,x2-y2,xy);
    ! column offset of shell l is l*l (s:1, p:2-4, d:5-9)
    yc(:,1)=ylm(:,1); yc(:,2)=-ylm(:,3); yc(:,3)=-ylm(:,4); yc(:,4)=ylm(:,2)
    yc(:,5)=ylm(:,5); yc(:,6)=-ylm(:,6); yc(:,7)=-ylm(:,7); yc(:,8)=ylm(:,8); yc(:,9)=ylm(:,9)
    call interp_atwfc(n,qlen,nwfcm,chiq)
    cache(s)%phi=(0._dp,0._dp); cache(s)%chi=(0._dp,0._dp); cache(s)%u=(0._dp,0._dp); cache(s)%w=(0._dp,0._dp)
    do is=1,nsite
      associate(s_=>site(is))
      a0=s_%ao0
      do ish=1,s_%nshell
        ga=s_%shell_grp(ish); l=s_%shell_l(ish); nt=ityp(s_%atom(ga))
        lph=(0._dp,-1._dp)**l
        if(s_%center_frozen) then
          p=s_%cref(:,ga)
        else
          p=pos_bohr(s_%atom(ga),s_%img(:,ga))
        endif
        do i=1,n
          sk(i)=lph*exp(cmplx(0._dp,-dot_product(q(:,i),p),dp))*chiq(i,s_%shell_chi(ish),nt)
        enddo
        do m=1,ofe_nlm(l)
          col=a0+s_%shell_start(ish)+m-1
          cache(s)%phi(1:n,col)=sk(1:n)*yc(1:n,l*l+m)
        enddo
      enddo
      call ofe_group_build(n,lda,s_%nao,cache(s)%phi(:,a0+1:a0+s_%nao),cache(s)%chi(:,a0+1:a0+s_%nao), &
                           cache(s)%t(1:s_%nao,1:s_%nao,is),cache(s)%eval(1:s_%nao,is), &
                           cache(s)%evec(1:s_%nao,1:s_%nao,is),cache(s)%u(:,a0+1:a0+s_%nao), &
                           size(vkb,1),vkb,bt,red_c,info,cond)
      if(info/=0) call fatal('Group S metric non-Hermitian, ill-conditioned or Lowdin check failed',abs(info))
      s_%condmax=max(s_%condmax,cond)
      allocate(cl(s_%nao,s_%nch))
      cl=cmplx(s_%clab(1:s_%nao,1:s_%nch),0._dp,dp)
      call zgemm('N','N',n,s_%nch,s_%nao,(1._dp,0._dp),cache(s)%u(1,a0+1),lda,cl,s_%nao,(0._dp,0._dp), &
                 cache(s)%w(1,s_%col0+1),lda)
      deallocate(cl)
      end associate
    enddo
    cache(s)%ready=.true.; cache(s)%npw=n
    deallocate(q,q2,qlen,ylm,yc,chiq,sk)
  end subroutine ofe_prepare

  ! ======================================================================== H psi
  subroutine ofe_hpsi(lda,n,m,psi,hpsi)
    use wvfct, only: current_k
    use lsda_mod, only: isk
    use klist, only: nks, ngk
    integer, intent(in) :: lda, n, m
    complex(dp), intent(in) :: psi(lda,m)
    complex(dp), intent(inout) :: hpsi(lda,m)
    integer :: s
    if(.not.active .or. .not.nonzero) return
    if(current_k<1 .or. current_k>nks) call fatal('h_psi without a valid current_k',1)
    if(n/=ngk(current_k)) call fatal('h_psi n differs from ngk(current_k)',1)
    call start_clock('ofe_hpsi')
    call ofe_prepare(current_k,n,lda)
    h_apply_count=h_apply_count+1
    s=kslot(current_k)
    call ofe_apply_stack(n,lda,m,nchtot,cache(s)%w,dall(:,:,isk(current_k)),psi,hpsi,red_c)
    call stop_clock('ofe_hpsi')
  end subroutine ofe_hpsi

#if defined(__OFE_GPU)
  ! ======================================================================== H psi, GPU G1 host bridge
  subroutine ofe_hpsi_gpu(lda,n,m,psi,hpsi)
    ! Called at the end of h_psi__gpu, where psi and hpsi are device-resident (present).  G1 route:
    ! psi device->host, dH = W D W^H psi on the host with the unchanged CPU kernel (ofe_hpsi),
    ! dH host->device and added there.  hpsi is never copied back to the host.  Host vkb is stale in the
    ! h_psi path (c_bands: init_us_2(...,.true.) updates only the device), so it is pulled only when the
    ! k cache is about to be rebuilt by ofe_prepare (the only host consumer of vkb in ofe_hpsi).
    use uspp, only: vkb, nkb
    use wvfct, only: current_k
    use klist, only: nks
    integer, intent(in) :: lda, n, m
    complex(dp), intent(in) :: psi(lda,m)
    complex(dp), intent(inout) :: hpsi(lda,m)
    complex(dp), allocatable :: dh(:,:)
    integer :: i, j
    if(.not.active .or. .not.nonzero) return
    if(m<1) return
    if(current_k<1 .or. current_k>nks) call fatal('h_psi_gpu without a valid current_k',1)
    call start_clock('ofe_gpu_sync')
    call ofe_refresh()                         ! tau changed -> every k slot not ready
    if(.not.cache(kslot(current_k))%ready .and. nkb>0) then
      !$acc update self(vkb)
    endif
    !$acc update self(psi(1:lda,1:m))
    allocate(dh(lda,m))
    dh=(0._dp,0._dp)
    call ofe_hpsi(lda,n,m,psi,dh)              ! host: dh = W D W^H psi (V1.1 kernel, counts h_apply_count)
    !$acc parallel loop collapse(2) present(hpsi) copyin(dh)
    do j=1,m
      do i=1,n
        hpsi(i,j)=hpsi(i,j)+dh(i,j)
      enddo
    enddo
    deallocate(dh)
    h_apply_count_gpu=h_apply_count_gpu+1
    call stop_clock('ofe_gpu_sync')
  end subroutine ofe_hpsi_gpu
#endif

  ! ======================================================================== forces and diagnostics
  subroutine ofe_forces(force)
    use ions_base, only: nat
    use lsda_mod, only: nspin, isk, current_spin
    use wvfct, only: npwx, nbnd, wg, current_k
    use klist, only: nks, ngk, igk_k, xk, nkstot
    use io_files, only: nwordwfc, iunwfc
    use io_global, only: ionode, stdout
    use buffers, only: get_buffer
    use wavefunctions, only: evc
    use uspp, only: vkb
    use uspp_init, only: init_us_2
    use gvect, only: g
    use cell_base, only: tpiba
    use mp, only: mp_sum, mp_max
    use mp_pools, only: inter_pool_comm
    use control_flags, only: istep
#if defined(__OFE_GPU)
    use control_flags, only: use_gpu
#endif
    use symm_base, only: time_reversal
    real(dp), intent(inout) :: force(3,nat)
    complex(dp), allocatable :: c(:,:), cs(:,:), nmat(:,:,:,:), nch_(:,:), nwork(:,:), zwork(:)
    real(dp), allocatable :: q(:,:), fphi_s(:,:,:), fphid_s(:,:,:), faug_s(:,:,:), ffr_s(:,:,:), ftr_s(:,:)
    real(dp), allocatable :: fcol(:,:), faug(:,:), fexpl(:,:), ev(:), rwork(:)
    real(dp) :: eext(2,maxsite), etot, ftr(3), sumerr, frsum, imax, dedv, fmax, nmin, nmax_, sproj(3)
    integer :: ik, n, i, is, sp, oldk, j, k, p, io, nao, nchn, a0, panc(3), s, ip, nsupp, info
    integer :: oldspin, nchmax
    if(.not.active) return
    call ofe_refresh()
    if(nonzero .and. h_apply_count<1) call fatal('No Hamiltonian field applications observed',1)
#if defined(__OFE_GPU)
    if(use_gpu .and. nonzero .and. h_apply_count_gpu<1) &
      call fatal('GPU run: no field applications through h_psi_gpu observed (field would be skipped)',1)
    if(use_gpu .and. nks==1) then
      !$acc update self(evc) if_present
    endif
#endif
    call start_clock('ofe_force')
    allocate(c(naotot,nbnd),nmat(maxao,maxao,2,nsite))
    allocate(fphi_s(3,nat,nsite),fphid_s(3,nat,nsite),faug_s(3,nat,nsite),ffr_s(3,nat,nsite),ftr_s(3,nsite))
    allocate(fcol(3,maxao),faug(3,nat),fexpl(3,nat))
    nmat=(0._dp,0._dp); fphi_s=0; fphid_s=0; faug_s=0; ffr_s=0; ftr_s=0
    oldk=current_k
    if(print_hch) then
      oldspin=current_spin
      nchmax=maxval(site(1:nsite)%nch)
      allocate(hch_acc(maxch,maxch,2,nsite),hch_fac(maxch,maxch,2,nsite))
      hch_acc=(0._dp,0._dp); hch_fac=(0._dp,0._dp); hch_wsum=0; hch_low=0; hch_rq=0
      hch_nrq=min(hch_nrq_max,nbnd)
      if(print_hch_k) then
        allocate(hch_kbuf(nchmax,nchmax,nsite,nkstot),hch_kspin(nkstot))
        hch_kbuf=(0._dp,0._dp); hch_kspin=0
      endif
    endif
    do ik=1,nks
      n=ngk(ik); current_k=ik; sp=isk(ik)
      if(nks>1) call get_buffer(evc,nwordwfc,iunwfc,ik)
      call init_us_2(n,igk_k(1,ik),xk(1,ik),vkb)
      call ofe_prepare(ik,n,npwx)
      s=kslot(ik)
      call ofe_project(n,npwx,naotot,nbnd,size(evc,1),cache(s)%u,evc,c,red_c)
      allocate(q(3,n))
      do i=1,n
        q(:,i)=tpiba*(g(:,igk_k(i,ik))+xk(:,ik))
      enddo
      do is=1,nsite
        nao=site(is)%nao; a0=site(is)%ao0
        allocate(cs(nao,nbnd))
        cs=c(a0+1:a0+nao,:)
        call ofe_nmat_acc(nao,nbnd,cs,wg(:,ik),nmat(1:nao,1:nao,sp,is))
        if(nonzero .and. any(abs(site(is)%dch(1:site(is)%nch,1:site(is)%nch,sp))>0._dp)) then
          call ofe_force_k(n,npwx,nao,nbnd,q,cache(s)%phi(:,a0+1:a0+nao),cache(s)%chi(:,a0+1:a0+nao), &
                           cache(s)%t(1:nao,1:nao,is),cache(s)%eval(1:nao,is),cache(s)%evec(1:nao,1:nao,is), &
                           cache(s)%u(:,a0+1:a0+nao),size(evc,1),evc,wg(:,ik),cs,site(is)%dao(1:nao,1:nao,sp), &
                           size(vkb,1),vkb,bt,fcol(:,1:nao),faug,ftr,red_c)
          do i=1,nao
            if(site(is)%center_frozen) then
              fphid_s(:,site(is)%owner(i),is)=fphid_s(:,site(is)%owner(i),is)+fcol(:,i)
            else
              fphi_s(:,site(is)%owner(i),is)=fphi_s(:,site(is)%owner(i),is)+fcol(:,i)
            endif
          enddo
          faug_s(:,:,is)=faug_s(:,:,is)+faug
          ftr_s(:,is)=ftr_s(:,is)+ftr
        endif
        deallocate(cs)
      enddo
      if(print_hch) call hch_k(ik,n,s,sp)
      deallocate(q)
    enddo
    call mp_sum(fphi_s,inter_pool_comm); call mp_sum(fphid_s,inter_pool_comm)
    call mp_sum(faug_s,inter_pool_comm); call mp_sum(ftr_s,inter_pool_comm)
    call mp_sum(nmat,inter_pool_comm)
    do is=1,nsite
      call mp_max(site(is)%condmax,inter_pool_comm)
    enddo
    if(print_hch) then
      call mp_sum(hch_acc,inter_pool_comm); call mp_sum(hch_fac,inter_pool_comm); call mp_sum(hch_wsum,inter_pool_comm)
      call mp_max(hch_low,inter_pool_comm); call mp_max(hch_rq,inter_pool_comm)
      if(print_hch_k) then
        call mp_sum(hch_kbuf,inter_pool_comm); call mp_sum(hch_kspin,inter_pool_comm)
      endif
    endif
    ! frame term -Re Tr(N dD_ao/dR) on the anchors (moving frame only)
    eext=0
    do is=1,nsite
      nao=site(is)%nao
      do sp=1,nspin
        eext(sp,is)=real(sum(site(is)%dao(1:nao,1:nao,sp)*transpose(nmat(1:nao,1:nao,sp,is))),dp)
      enddo
      if(site(is)%mode/=2 .or. .not.nonzero) cycle
      panc=[site(is)%anc_atom(2),site(is)%anc_atom(3),site(is)%anc_atom(1)]
      do sp=1,nspin
        do p=1,3
          if(p==2 .and. .not.site(is)%use_x) cycle
          do k=1,3
            ffr_s(k,panc(p),is)=ffr_s(k,panc(p),is) &
              -real(sum(nmat(1:nao,1:nao,sp,is)*transpose(site(is)%ddao(1:nao,1:nao,k,p,sp))),dp)
          enddo
        enddo
      enddo
    enddo
    etot=sum(eext)
    fexpl=0
    do is=1,nsite
      fexpl=fexpl+fphi_s(:,:,is)+faug_s(:,:,is)+ffr_s(:,:,is)
    enddo
    force=force+fexpl
    fexpl_last=fexpl
    raw_seen=.false.
    if(ionode) then
      open(newunit=io,file=trim(history),status='unknown',position='append',action='write')
      write(io,'(a,i8)') 'OFE_BEGIN_STEP ',istep
      write(io,'(a,i12)') 'OFE_H_APPLY_COUNT ',h_apply_count
#if defined(__OFE_GPU)
      write(io,'(a,i12,a,l2)') 'OFE_H_APPLY_COUNT_GPU ',h_apply_count_gpu,' use_gpu',use_gpu
#endif
      imax=0; sumerr=0; frsum=0
      allocate(nwork(maxch,maxch),zwork(64*maxch),ev(maxch),rwork(3*maxch))
      do is=1,nsite
        associate(s_=>site(is))
        nao=s_%nao; nchn=s_%nch
        write(io,'(a,i4,1x,a,1x,a,es12.4)') 'OFE_SITE_STEP site tag kind cond_max ',is,trim(s_%tag),trim(s_%kind), &
          s_%condmax
        do j=1,3
          write(io,'(a,2i4,3es25.16)') 'OFE_FRAME site col ',is,j,s_%q(:,j)
        enddo
        allocate(nch_(nchn,nchn))
        do sp=1,nspin
          nch_=matmul(transpose(cmplx(s_%clab(1:nao,1:nchn),0._dp,dp)), &
                      matmul(nmat(1:nao,1:nao,sp,is),cmplx(s_%clab(1:nao,1:nchn),0._dp,dp)))
          imax=max(imax,maxval(abs(aimag(nch_))))
          do j=1,nchn
            write(io,'(a,3i4,*(es25.16))') 'OFE_N site spin row ',is,sp,j,real(nch_(j,:),dp)
          enddo
          if(.not.time_reversal) then
            do j=1,nchn
              write(io,'(a,3i4,*(es25.16))') 'OFE_N_IM site spin row ',is,sp,j,aimag(nch_(j,:))
            enddo
          endif
          if(print_hch) call write_hch(io,is,sp)
          if(print_nao) then
            do j=1,nao
              write(io,'(a,3i4,*(es25.16))') 'OFE_N_AO site spin row ',is,sp,j,real(nmat(j,1:nao,sp,is),dp)
            enddo
          endif
          nwork(1:nchn,1:nchn)=nch_
          if(time_reversal) nwork(1:nchn,1:nchn)=cmplx(real(nch_,dp),0._dp,dp)
          nwork(1:nchn,1:nchn)=0.5_dp*(nwork(1:nchn,1:nchn)+transpose(conjg(nwork(1:nchn,1:nchn))))
          call zheev('N','U',nchn,nwork,maxch,ev,zwork,size(zwork),rwork,info)
          if(info/=0) call fatal('Occupation eigenvalue calculation failed',info)
          nmin=minval(ev(1:nchn)); nmax_=maxval(ev(1:nchn))
          write(io,'(a,2i4,3es20.11)') 'OFE_N_EIG_MIN_MAX_TRACE site spin ',is,sp,nmin,nmax_, &
            sum([(real(nch_(j,j),dp),j=1,nchn)])
          write(io,'(a,2i4,es25.16)') 'OFE_EEXT site spin Ry ',is,sp,eext(sp,is)
          do ip=1,s_%npar
            if(s_%pspin(ip)/=0 .and. s_%pspin(ip)/=sp) cycle
            dedv=real(sum(s_%pb(1:nchn,1:nchn,ip)*transpose(nch_)),dp)
            write(io,'(a,i4,1x,a,i3,2es25.16)') 'OFE_DEDV site name spin value_eV dE/dV(e) ',is, &
              trim(s_%pname(ip)),sp,s_%pval(ip),dedv
            write(stdout,'(a,i4,1x,a,i3,2es25.16)') 'OFE_DEDV site name spin value_eV dE/dV(e) ',is, &
              trim(s_%pname(ip)),sp,s_%pval(ip),dedv
          enddo
        enddo
        deallocate(nch_)
        do j=1,nat
          write(io,'(a,2i6,12es20.11)') 'OFE_FORCE_SPLIT site atom phi aug frame cder ',is,j,fphi_s(:,j,is), &
            faug_s(:,j,is),ffr_s(:,j,is),0._dp,0._dp,0._dp
        enddo
        if(s_%center_frozen) then
          do j=1,nat
            if(any(fphid_s(:,j,is)/=0._dp)) write(io,'(a,2i6,3es20.11)') &
              'OFE_FORCE_PHI_FROZEN_DIAG site atom (not applied) ',is,j,fphid_s(:,j,is)
          enddo
        endif
        sproj=sum(fphi_s(:,:,is)+fphid_s(:,:,is)+faug_s(:,:,is),dim=2)
        sumerr=max(sumerr,maxval(abs(sproj-ftr_s(:,is))))
        frsum=max(frsum,maxval(abs(sum(ffr_s(:,:,is),dim=2))))
        write(io,'(a,i4,9es25.16)') 'OFE_SUMRULE site sum_proj commutator sum_frame ',is,sproj,ftr_s(:,is), &
          sum(ffr_s(:,:,is),dim=2)
        end associate
      enddo
      deallocate(nwork,zwork,ev,rwork)
      if(time_reversal) then
        write(io,'(a,es14.6,a)') 'OFE_N_IMAG_RAW_MAX ',imax,' TR_REDUCED_K_ARTEFACT'
      else
        write(io,'(a,es14.6,a)') 'OFE_N_IMAG_RAW_MAX ',imax,' FULL_MESH_PHYSICAL'
      endif
      if(print_hch) write(io,'(a,i4,es14.6)') 'OFE_DIAG_HCH_RAYLEIGH nbands maxerr_Ry ',hch_nrq,hch_rq
      write(io,'(a,es25.16)') 'OFE_EEXT_TOTAL_RY_ALREADY_IN_EBAND ',etot
      do j=1,nat
        write(io,'(a,i6,3es25.16)') 'OFE_FORCE_EXPLICIT_TOTAL ',j,fexpl(:,j)
      enddo
      write(io,'(a,3es25.16)') 'OFE_FORCE_EXPLICIT_SUM ',sum(fexpl,dim=2)
      close(io)
      write(stdout,'(/,a,a)') '     OFE summary  ',ofe_version
      write(stdout,'(a,es25.16)') 'OFE_EEXT_TOTAL_RY_ALREADY_IN_EBAND ',etot
      write(stdout,'(a,es14.6,a,es14.6)') 'OFE_SUMRULE_MAXERR ',sumerr,'  OFE_FRAME_SUM_MAX ',frsum
      write(stdout,'(a,3es16.8)') 'OFE_FORCE_EXPLICIT_SUM_RY_BOHR ',sum(fexpl,dim=2)
      if(print_hch) write(stdout,'(a,es12.4,a,i3,a,es12.4)') 'OFE_HCH written; lowdin_err ',maxval(hch_low(1:nsite)), &
        '  rayleigh(nb=',hch_nrq,') ',hch_rq
      nsupp=0; fmax=0
      do j=1,nat
        if(maxval(abs(fexpl(:,j)))>=fprint_min) then
          write(stdout,'(a,i6,3es20.10)') 'OFE_FORCE_EXPLICIT_RY_BOHR ',j,fexpl(:,j)
        else
          nsupp=nsupp+1; fmax=max(fmax,maxval(abs(fexpl(:,j))))
        endif
      enddo
      write(stdout,'(a,i6,es12.4)') 'OFE_FORCE_EXPLICIT_SUPPRESSED count max ',nsupp,fmax
      flush(stdout)
    endif
    current_k=oldk
    if(oldk>=1 .and. oldk<=nks) then
      call init_us_2(ngk(oldk),igk_k(1,oldk),xk(1,oldk),vkb)
      if(nks>1) call get_buffer(evc,nwordwfc,iunwfc,oldk)
    endif
    if(print_hch) then
      current_spin=oldspin
      if(oldk>=1 .and. oldk<=nks) call g2_kin(oldk)
      deallocate(hch_acc,hch_fac)
      if(print_hch_k) deallocate(hch_kbuf,hch_kspin)
    endif
#if defined(__OFE_GPU)
    if(use_gpu) then
      !$acc update device(evc) if_present
    endif
#endif
    deallocate(c,nmat,fphi_s,fphid_s,faug_s,ffr_s,ftr_s,fcol,faug,fexpl)
    call stop_clock('ofe_force')
  end subroutine ofe_forces

  ! ======================================================================== V1.2 native channel Hamiltonian
  subroutine hch_k(ik,n,s,sp)
    ! OFE_HCH for one local k point.  On entry (ofe_forces k loop): current_k = ik, vkb = beta(ik) on host
    ! and device (init_us_2 default), evc = psi(ik) on the host, cache(s) ready.
    !   phit(:,a0+1:a0+nao) = Phi T: group S-Lowdin AOs (U = S phit, U^H phit = T G T = 1)
    !   phit(:,naotot+1:naotot+nrq) = the lowest nrq bands (Rayleigh diagnostic <psi|H|psi> = et)
    !   H phit by ONE h_psi (CPU) or h_psi_gpu (GPU) call, H = current KS operator INCLUDING the OFE field
    !   H_ao = phit^H H phit (intra_bgrp_comm reduction); H_ch = C_lab^T H_ao C_lab (C_lab real)
    !   exact field part in the same basis: C_lab^T (W^H phit)^H Dall (W^H phit) C_lab (= D for an isolated site)
    ! State: current_spin and g2kin are set for ik here and restored by ofe_forces; becp (unallocated in
    ! forces: diag_bands and force_us free it) is allocated for ncol columns and freed; h_apply_count and
    ! h_apply_count_gpu are restored, so HCH applications are never counted as SCF field applications.
    use wvfct, only: npwx, et
    use klist, only: wk, nkstot
    use lsda_mod, only: lsda, isk, current_spin
    use uspp, only: nkb
    use becmod, only: becp, allocate_bec_type_acc, deallocate_bec_type_acc
    use mp_bands, only: intra_bgrp_comm
    use wavefunctions, only: evc
#if defined(__OFE_GPU)
    use control_flags, only: use_gpu
#endif
    integer, intent(in) :: ik, n, s, sp
    integer, external :: global_kpoint_index
    external :: h_psi, g2_kin
#if defined(__OFE_GPU)
    external :: h_psi_gpu
#endif
    complex(dp), parameter :: z1=(1._dp,0._dp), z0=(0._dp,0._dp)
    complex(dp), allocatable :: phit(:,:), hphit(:,:), hao(:,:), uu(:,:), mm(:,:), rq(:,:)
    complex(dp), allocatable :: cl(:,:), fao(:,:), hc(:,:), fc(:,:)
    integer :: ncol, nrq, is, nao, a0, nchn, i, ib, keep_cpu, keep_gpu, ikg
    call start_clock('ofe_hch')
    if(size(cache(s)%phi,1)/=npwx .or. size(evc,1)/=npwx) call fatal('OFE_HCH: leading dimension differs from npwx',1)
    nrq=hch_nrq; ncol=naotot+nrq
    allocate(phit(npwx,ncol),hphit(npwx,ncol))
    phit=z0; hphit=z0
    do is=1,nsite
      nao=site(is)%nao; a0=site(is)%ao0
      call zgemm('N','N',n,nao,nao,z1,cache(s)%phi(1,a0+1),npwx,cache(s)%t(1,1,is),maxao,z0,phit(1,a0+1),npwx)
    enddo
    if(nrq>0) phit(1:n,naotot+1:ncol)=evc(1:n,1:nrq)
    ! Lowdin identity U^H phit = 1 on each site block (diagnostic of the exact field subtraction)
    allocate(uu(naotot,naotot))
    call zgemm('C','N',naotot,naotot,n,z1,cache(s)%u,npwx,phit,npwx,z0,uu,naotot)
    call red_c(uu)
    do is=1,nsite
      nao=site(is)%nao; a0=site(is)%ao0
      do i=1,nao
        uu(a0+i,a0+i)=uu(a0+i,a0+i)-z1
      enddo
      hch_low(is)=max(hch_low(is),maxval(abs(uu(a0+1:a0+nao,a0+1:a0+nao))))
    enddo
    deallocate(uu)
    ! one Hamiltonian application for all sites and the Rayleigh bands
    if(lsda) current_spin=isk(ik)
    call g2_kin(ik)
    if(allocated(becp%k) .or. allocated(becp%r) .or. allocated(becp%nc)) &
      call fatal('OFE_HCH: becp unexpectedly allocated in forces',1)
    call allocate_bec_type_acc(nkb,ncol,becp,intra_bgrp_comm)
    keep_cpu=h_apply_count; keep_gpu=h_apply_count_gpu
#if defined(__OFE_GPU)
    if(use_gpu) then
      !$acc data copyin(phit) copyout(hphit)
      call h_psi_gpu(npwx,n,ncol,phit,hphit)
      !$acc end data
    else
      call h_psi(npwx,n,ncol,phit,hphit)
    endif
#else
    call h_psi(npwx,n,ncol,phit,hphit)
#endif
    h_apply_count=keep_cpu; h_apply_count_gpu=keep_gpu
    call deallocate_bec_type_acc(becp)
    ! projections (one ZGEMM + one reduction each)
    allocate(hao(naotot,naotot),mm(nchtot,naotot),rq(max(nrq,1),1))
    call zgemm('C','N',naotot,naotot,n,z1,phit,npwx,hphit,npwx,z0,hao,naotot)
    call red_c(hao)
    call zgemm('C','N',nchtot,naotot,n,z1,cache(s)%w,npwx,phit,npwx,z0,mm,nchtot)
    call red_c(mm)
    rq=z0
    do ib=1,nrq
      rq(ib,1)=sum(conjg(phit(1:n,naotot+ib))*hphit(1:n,naotot+ib))
    enddo
    call red_c(rq)
    do ib=1,nrq
      hch_rq=max(hch_rq,abs(real(rq(ib,1),dp)-et(ib,ik)))
    enddo
    hch_wsum(sp)=hch_wsum(sp)+wk(ik)
    ikg=0
    if(print_hch_k) then
      ikg=global_kpoint_index(nkstot,ik)
      if(ikg<1 .or. ikg>nkstot) call fatal('OFE_HCH: global k index out of range',ikg)
      hch_kspin(ikg)=sp
    endif
    do is=1,nsite
      nao=site(is)%nao; a0=site(is)%ao0; nchn=site(is)%nch
      allocate(cl(nao,nchn),fao(nao,nao),hc(nchn,nchn),fc(nchn,nchn))
      cl=cmplx(site(is)%clab(1:nao,1:nchn),0._dp,dp)
      hc=matmul(transpose(cl),matmul(hao(a0+1:a0+nao,a0+1:a0+nao),cl))
      fao=matmul(transpose(conjg(mm(:,a0+1:a0+nao))),matmul(dall(:,:,sp),mm(:,a0+1:a0+nao)))
      fc=matmul(transpose(cl),matmul(fao,cl))
      hch_acc(1:nchn,1:nchn,sp,is)=hch_acc(1:nchn,1:nchn,sp,is)+wk(ik)*hc
      hch_fac(1:nchn,1:nchn,sp,is)=hch_fac(1:nchn,1:nchn,sp,is)+wk(ik)*fc
      if(print_hch_k) hch_kbuf(1:nchn,1:nchn,is,ikg)=hc
      deallocate(cl,fao,hc,fc)
    enddo
    deallocate(phit,hphit,hao,mm,rq)
    call stop_clock('ofe_hch')
  end subroutine hch_k

  subroutine write_hch(io,is,sp)
    ! OFE_HCH records of one site and spin (shared interface spec V1.2): Ry, es25.16, written right after
    ! the OFE_N (and OFE_N_IM) rows of that site and spin.  Ionode only; arrays already pool-reduced.
    use symm_base, only: time_reversal
    use klist, only: nkstot
    integer, intent(in) :: io, is, sp
    complex(dp), allocatable :: hb(:,:), fb(:,:)
    integer :: n_, j, ikg
    real(dp) :: herr, xerr, imx
    n_=site(is)%nch
    if(hch_wsum(sp)<=0._dp) call fatal('OFE_HCH: zero total k weight for a spin channel',sp)
    allocate(hb(n_,n_),fb(n_,n_))
    hb=hch_acc(1:n_,1:n_,sp,is)/hch_wsum(sp)
    fb=hch_fac(1:n_,1:n_,sp,is)/hch_wsum(sp)
    do j=1,n_
      write(io,'(a,3i4,*(es25.16))') 'OFE_HCH ',is,sp,j,real(hb(j,:),dp)
    enddo
    do j=1,n_
      write(io,'(a,3i4,*(es25.16))') 'OFE_HCH_FIELD ',is,sp,j,real(site(is)%dch(j,1:n_,sp),dp)
    enddo
    if(any(aimag(site(is)%dch(1:n_,1:n_,sp))/=0._dp)) then
      do j=1,n_
        write(io,'(a,3i4,*(es25.16))') 'OFE_HCH_FIELD_IM ',is,sp,j,aimag(site(is)%dch(j,1:n_,sp))
      enddo
    endif
    if(.not.time_reversal) then
      do j=1,n_
        write(io,'(a,3i4,*(es25.16))') 'OFE_HCH_IM ',is,sp,j,aimag(hb(j,:))
      enddo
    endif
    imx=maxval(abs(aimag(hb)))
    if(time_reversal) then
      write(io,'(a,2i4,es14.6,a)') 'OFE_HCH_IMAG_RAW_MAX ',is,sp,imx,' TR_REDUCED_K_ARTEFACT'
    else
      write(io,'(a,2i4,es14.6,a)') 'OFE_HCH_IMAG_RAW_MAX ',is,sp,imx,' FULL_MESH_PHYSICAL'
    endif
    if(print_hch_k) then
      do ikg=1,nkstot
        if(hch_kspin(ikg)/=sp) cycle
        do j=1,n_
          write(io,'(a,2i4,i6,i4,*(es25.16))') 'OFE_HCH_K ',is,sp,ikg,j,real(hch_kbuf(j,1:n_,is,ikg),dp), &
            aimag(hch_kbuf(j,1:n_,is,ikg))
        enddo
      enddo
    endif
    herr=maxval(abs(hb-transpose(conjg(hb))))
    if(time_reversal) then
      xerr=maxval(abs(real(fb,dp)-real(site(is)%dch(1:n_,1:n_,sp),dp)))
    else
      xerr=maxval(abs(fb-site(is)%dch(1:n_,1:n_,sp)))
    endif
    write(io,'(a,2i4,3es14.6)') 'OFE_DIAG_HCH site spin herm_Ry xsite_Ry lowdin ',is,sp,herr,xerr,hch_low(is)
    deallocate(hb,fb)
  end subroutine write_hch

  subroutine ofe_record_raw_component(ip,f)
    integer, intent(in) :: ip
    real(dp), intent(in) :: f(:)
    if(.not.active) return
    if(ip<1 .or. ip>3 .or. size(f)/=size(raw_force,2)) call fatal('Raw force dimensions',1)
    raw_force(ip,:)=f; raw_seen(ip)=.true.
  end subroutine ofe_record_raw_component

  subroutine ofe_finish_raw()
    use io_global, only: ionode, stdout
    use ions_base, only: nat
    integer :: j, io
    if(.not.active) return
    if(.not.all(raw_seen)) call fatal('Incomplete raw force components',1)
    if(.not.ionode) return
    open(newunit=io,file=trim(history),status='unknown',position='append',action='write')
    write(io,'(a,3es25.16)') 'OFE_RAW_FORCE_SUM ',sum(raw_force,dim=2)
    do j=1,nat
      write(io,'(a,i6,6es25.16)') 'OFE_FORCE_RAW atom pos_bohr raw ',j,pos_bohr(j,[0,0,0]),raw_force(:,j)
    enddo
    do j=1,nat
      write(io,'(a,i6,3es25.16)') 'OFE_FORCE_NATIVE ',j,raw_force(:,j)-fexpl_last(:,j)
    enddo
    close(io)
    write(stdout,'(a,3es25.16)') 'OFE_RAW_FORCE_SUM_RY_BOHR ',sum(raw_force,dim=2)
    flush(stdout)
  end subroutine ofe_finish_raw

  subroutine ofe_record_final(force)
    ! final QE force (net force removed, symmetrised, external forces), before if_pos masking
    use io_global, only: ionode
    use control_flags, only: istep
    use ions_base, only: nat
    real(dp), intent(in) :: force(3,nat)
    integer :: j, io
    if(.not.active .or. .not.ionode) return
    open(newunit=io,file=trim(history),status='unknown',position='append',action='write')
    do j=1,nat
      write(io,'(a,2i6,3es25.16)') 'OFE_FORCE_FINAL_QE ',istep,j,force(:,j)
    enddo
    write(io,'(a,i8)') 'OFE_END_STEP ',istep
    close(io)
    if(nonzero) call print_clock('ofe_hpsi')
    call print_clock('ofe_force')
    if(print_hch) call print_clock('ofe_hch')
#if defined(__OFE_GPU)
    if(nonzero) call print_clock('ofe_gpu_sync')
#endif
  end subroutine ofe_record_final
end module orbital_field_engine
