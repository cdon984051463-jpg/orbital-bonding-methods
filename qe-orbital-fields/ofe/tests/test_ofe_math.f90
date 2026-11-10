! Standalone test of the OFE_V1 production kernels (ofe_math.f90), no QE needed.
! Toy periodic plane-wave model with Gaussian AOs/betas (analytic Fourier transforms), random
! S-orthonormal psi on two (k, spin) entries; for six site configurations covering the four designs
! (RU_D, O_P_BOND with helper axis, RU_O_PI joint d+p with a lattice image and complex D,
! IAO_CHANNEL joint group with a Lowdin-only shell, frozen centre, frozen / global frames,
! spin-resolved D with shared C): analytic explicit force vs central differences on EVERY atom,
! translation identities, atom classes, stacked apply, hermiticity, HF linearity, axial check.
program test_ofe_math
  use ofe_math
  implicit none
  integer, parameter :: dp=ofe_dp, nat=6, nk=2, nb=7, mao=16
  real(dp), parameter :: pi=3.141592653589793238_dp, ecut=9._dp
  real(dp) :: alat(3,3), brec(3,3), omega, kfrac(3,nk)
  integer :: npw(nk), ldq
  real(dp), allocatable :: qv(:,:,:)
  integer :: spec(nat)=[1,2,2,3,2,4], nhs(4)=[6,4,3,0]
  real(dp) :: qqs(6,6,4), tau0(3,nat), w(nb,nk)
  type(ofe_beta_t) :: bt
  complex(dp), allocatable :: psi(:,:,:)
  type case_t
    character(len=28) :: name=' '
    integer :: nao=0, nch=0, nshell=0, mode=2
    integer :: shell_l(4)=0, shell_start(4)=0, shell_atom(4)=0, shell_img(3,4)=0
    real(dp) :: shell_alpha(4)=0
    real(dp) :: cloc(mao,mao)=0
    complex(dp) :: dch(mao,mao,2)=0
    integer :: anc(3)=0, anc_img(3,3)=0
    logical :: use_x=.true., center_frozen=.false.
    real(dp) :: helper(3)=0, qref(3,3)=0, cref(3,4)=0
  end type case_t
  type(case_t) :: cs(6)
  integer :: nfail, ic
  nfail=0
  call setup()
  call check_inv_sqrt(); call check_rot(); call check_frame(); call check_ylm()
  call make_cases()
  do ic=1,6
    call check_case(cs(ic))
  enddo
  call check_stack_and_hermitian()
  call check_axial()
  if(nfail==0) then
    write(*,'(a)') 'ALL_TESTS_PASS'
  else
    write(*,'(a,i3)') 'TESTS_FAILED ',nfail
    stop 1
  endif
contains
  subroutine report(name,e,tol)
    character(len=*), intent(in) :: name
    real(dp), intent(in) :: e, tol
    if(e<=tol) then
      write(*,'(a,1x,a,es12.4,a,es9.1)') 'PASS',name,e,' tol',tol
    else
      write(*,'(a,1x,a,es12.4,a,es9.1)') 'FAIL',name,e,' tol',tol
      nfail=nfail+1
    endif
  end subroutine report

  subroutine noop(a)
    complex(dp), intent(inout) :: a(:,:)
    if(size(a)<0) a=0
  end subroutine noop

  subroutine rand_mat(a)
    real(dp), intent(out) :: a(:,:)
    real(dp) :: u(size(a,1),size(a,2)), v(size(a,1),size(a,2))
    call random_number(u); call random_number(v)
    a=sqrt(-2*log(max(u,1.e-300_dp)))*cos(2*pi*v)
  end subroutine rand_mat

  real(dp) function radial(l,alpha,q)
    integer, intent(in) :: l
    real(dp), intent(in) :: alpha, q
    real(dp) :: nrm
    nrm=1/sqrt(gamma(l+1.5_dp)/(2*(2*alpha)**(l+1.5_dp)))
    radial=nrm*(pi/alpha)**1.5_dp*(q/(2*alpha))**l*exp(-q*q/(4*alpha))/sqrt(omega)
  end function radial

  real(dp) function det3(a)
    real(dp), intent(in) :: a(3,3)
    det3=a(1,1)*(a(2,2)*a(3,3)-a(2,3)*a(3,2))-a(1,2)*(a(2,1)*a(3,3)-a(2,3)*a(3,1))+a(1,3)*(a(2,1)*a(3,2)-a(2,2)*a(3,1))
  end function det3

  function inv3(a) result(b)
    real(dp), intent(in) :: a(3,3)
    real(dp) :: b(3,3), d
    d=det3(a)
    b(1,1)=a(2,2)*a(3,3)-a(2,3)*a(3,2); b(1,2)=a(1,3)*a(3,2)-a(1,2)*a(3,3); b(1,3)=a(1,2)*a(2,3)-a(1,3)*a(2,2)
    b(2,1)=a(2,3)*a(3,1)-a(2,1)*a(3,3); b(2,2)=a(1,1)*a(3,3)-a(1,3)*a(3,1); b(2,3)=a(1,3)*a(2,1)-a(1,1)*a(2,3)
    b(3,1)=a(2,1)*a(3,2)-a(2,2)*a(3,1); b(3,2)=a(1,2)*a(3,1)-a(1,1)*a(3,2); b(3,3)=a(1,1)*a(2,2)-a(1,2)*a(2,1)
    b=b/d
  end function inv3

  subroutine orthonormalize(x)
    real(dp), intent(inout) :: x(:,:)
    integer :: i, j
    do j=1,size(x,2)
      do i=1,j-1
        x(:,j)=x(:,j)-dot_product(x(:,i),x(:,j))*x(:,i)
      enddo
      x(:,j)=x(:,j)/sqrt(sum(x(:,j)**2))
    enddo
  end subroutine orthonormalize

  function eye(n) result(a)
    integer, intent(in) :: n
    complex(dp) :: a(n,n)
    integer :: i
    a=0
    do i=1,n
      a(i,i)=1
    enddo
  end function eye

  subroutine setup()
    integer :: n1, n2, n3, nmax, ik, cnt, j, isd(8), nkb
    real(dp) :: g(3), m(6,6)
    isd=[(23*j+5,j=1,8)]
    call random_seed(put=isd)
    alat=reshape([7._dp,0.2_dp,0._dp, 0.3_dp,6.5_dp,-0.3_dp, 0._dp,0.4_dp,7.5_dp],[3,3])  ! columns a1,a2,a3
    omega=abs(det3(alat))
    brec=2*pi*transpose(inv3(alat))
    kfrac(:,1)=[0.13_dp,-0.21_dp,0.07_dp]; kfrac(:,2)=[0.31_dp,0.17_dp,-0.40_dp]
    nmax=5; ldq=0
    do ik=1,nk
      cnt=0
      do n1=-nmax,nmax; do n2=-nmax,nmax; do n3=-nmax,nmax
        g=matmul(brec,[n1+kfrac(1,ik),n2+kfrac(2,ik),n3+kfrac(3,ik)])
        if(sum(g*g)<ecut) cnt=cnt+1
      enddo; enddo; enddo
      npw(ik)=cnt; ldq=max(ldq,cnt)
    enddo
    allocate(qv(3,ldq,nk)); qv=0
    do ik=1,nk
      cnt=0
      do n1=-nmax,nmax; do n2=-nmax,nmax; do n3=-nmax,nmax
        g=matmul(brec,[n1+kfrac(1,ik),n2+kfrac(2,ik),n3+kfrac(3,ik)])
        if(sum(g*g)<ecut) then
          cnt=cnt+1; qv(:,cnt,ik)=g
        endif
      enddo; enddo; enddo
    enddo
    qqs=0
    do j=1,3
      call rand_mat(m(1:nhs(j),1:nhs(j)))
      qqs(1:nhs(j),1:nhs(j),j)=0.25_dp*matmul(m(1:nhs(j),1:nhs(j)),transpose(m(1:nhs(j),1:nhs(j))))/nhs(j)
      do n1=1,nhs(j)
        qqs(n1,n1,j)=qqs(n1,n1,j)-0.02_dp
      enddo
    enddo
    bt%nat=nat; bt%nhm=6
    allocate(bt%ofs(nat),bt%nh(nat),bt%us(nat),bt%qq(6,6,nat))
    nkb=0
    do j=1,nat
      bt%ofs(j)=nkb; bt%nh(j)=nhs(spec(j)); nkb=nkb+bt%nh(j); bt%us(j)=(nhs(spec(j))>0)
      bt%qq(:,:,j)=qqs(:,:,spec(j))
    enddo
    bt%nkb=nkb
    ! 1 Ru-like, 2 O bonded to 1 (stored one cell away along a3: image (0,0,-1) brings it back),
    ! 3 O (side), 4 X (PAW, unrelated to every site), 5 O (anchor), 6 atom without beta (unrelated)
    tau0=reshape([0.2_dp,0.1_dp,0.3_dp, 0.4_dp-0._dp,-0.3_dp,3.9_dp, -0.2_dp,0.5_dp,-3.4_dp, &
                  3.0_dp,2.8_dp,0.4_dp, 3.4_dp,-0.5_dp,0.6_dp, -2.6_dp,-2.9_dp,1.1_dp],[3,nat])
    tau0(:,2)=tau0(:,2)+alat(:,3)          ! stored position = bonded position + a3
    allocate(psi(ldq,nb,nk)); psi=0
    do ik=1,nk
      call make_psi(ik)
      call random_number(w(:,ik)); w(:,ik)=0.5_dp*w(:,ik)
    enddo
  end subroutine setup

  subroutine build_beta(ik,tau,beta)
    integer, intent(in) :: ik
    real(dp), intent(in) :: tau(3,nat)
    complex(dp), intent(out) :: beta(ldq,bt%nkb)
    integer :: j, i, col, l, ish, m, nl, nbl(4), bl(3,4)
    real(dp) :: y(5), qn, ba(3,4)
    complex(dp) :: ph
    nbl=[2,2,1,0]; bl=0; ba=0
    bl(1:2,1)=[0,2]; ba(1:2,1)=[1.6_dp,1.8_dp]
    bl(1:2,2)=[0,1]; ba(1:2,2)=[2.0_dp,1.7_dp]
    bl(1,3)=1; ba(1,3)=1.5_dp
    beta=0
    do j=1,nat
      col=bt%ofs(j)
      do ish=1,nbl(spec(j))
        l=bl(ish,spec(j)); nl=ofe_nlm(l)
        do i=1,npw(ik)
          qn=sqrt(sum(qv(:,i,ik)**2))
          call ofe_ylm_cart(l,qv(:,i,ik),y(1:nl))
          ph=(0._dp,-1._dp)**l*exp(cmplx(0._dp,-dot_product(qv(:,i,ik),tau(:,j)),dp))*radial(l,ba(ish,spec(j)),qn)
          do m=1,nl
            beta(i,col+m)=ph*y(m)
          enddo
        enddo
        col=col+nl
      enddo
    enddo
  end subroutine build_beta

  subroutine make_psi(ik)
    integer, intent(in) :: ik
    real(dp) :: a(ldq,nb), b(ldq,nb), e(nb)
    complex(dp) :: p(ldq,nb), sp(ldq,nb), g(nb,nb), t(nb,nb), ev(nb,nb)
    complex(dp), allocatable :: beta(:,:)
    integer :: info
    allocate(beta(ldq,bt%nkb))
    call rand_mat(a); call rand_mat(b)
    p=cmplx(a,b,dp); p(npw(ik)+1:,:)=0
    call build_beta(ik,tau0,beta)
    call ofe_s_apply(npw(ik),ldq,nb,p,sp,ldq,beta,bt,noop)
    g=matmul(transpose(conjg(p(1:npw(ik),:))),sp(1:npw(ik),:))
    call ofe_inv_sqrt(nb,g,t,e,ev,info)
    if(info/=0) stop 'psi metric'
    psi(:,:,ik)=matmul(p,t)
  end subroutine make_psi

  function center(c,tau,ish) result(p)
    type(case_t), intent(in) :: c
    real(dp), intent(in) :: tau(3,nat)
    integer, intent(in) :: ish
    real(dp) :: p(3)
    if(c%center_frozen) then
      p=c%cref(:,ish)
    else
      p=tau(:,c%shell_atom(ish))+matmul(alat,real(c%shell_img(:,ish),dp))
    endif
  end function center

  subroutine build_phi(c,ik,tau,phi)
    type(case_t), intent(in) :: c
    integer, intent(in) :: ik
    real(dp), intent(in) :: tau(3,nat)
    complex(dp), intent(out) :: phi(ldq,c%nao)
    integer :: ish, i, m, l, nl
    real(dp) :: y(5), qn, p(3)
    complex(dp) :: ph
    phi=0
    do ish=1,c%nshell
      l=c%shell_l(ish); nl=ofe_nlm(l); p=center(c,tau,ish)
      do i=1,npw(ik)
        qn=sqrt(sum(qv(:,i,ik)**2))
        call ofe_ylm_cart(l,qv(:,i,ik),y(1:nl))
        ph=(0._dp,-1._dp)**l*exp(cmplx(0._dp,-dot_product(qv(:,i,ik),p),dp))*radial(l,c%shell_alpha(ish),qn)
        do m=1,nl
          phi(i,c%shell_start(ish)+m-1)=ph*y(m)
        enddo
      enddo
    enddo
  end subroutine build_phi

  function apos(c,tau,k) result(p)
    type(case_t), intent(in) :: c
    real(dp), intent(in) :: tau(3,nat)
    integer, intent(in) :: k
    real(dp) :: p(3)
    p=tau(:,c%anc(k))+matmul(alat,real(c%anc_img(:,k),dp))
  end function apos

  subroutine frame_of(c,tau,q,dq)
    type(case_t), intent(in) :: c
    real(dp), intent(in) :: tau(3,nat)
    real(dp), intent(out) :: q(3,3), dq(3,3,3,2)
    real(dp) :: b(3)
    integer :: info, k
    q=0; dq=0
    do k=1,3
      q(k,k)=1
    enddo
    if(c%mode==1) q=c%qref
    if(c%mode/=2) return
    if(c%use_x) then
      b=apos(c,tau,3)-apos(c,tau,1)
    else
      b=c%helper
    endif
    call ofe_frame(apos(c,tau,2)-apos(c,tau,1),b,q,dq,info)
    if(info/=0) stop 'frame'
  end subroutine frame_of

  subroutine energy_force(c,tau,want,e,fphi,fphid,faug,ffr,ftr,nch_out)
    type(case_t), intent(in) :: c
    real(dp), intent(in) :: tau(3,nat)
    logical, intent(in) :: want
    real(dp), intent(out) :: e, fphi(3,nat), fphid(3,nat), faug(3,nat), ffr(3,nat), ftr(3)
    complex(dp), intent(out), optional :: nch_out(c%nch,c%nch,2)
    complex(dp) :: phi(ldq,c%nao), chi(ldq,c%nao), u(ldq,c%nao), t(c%nao,c%nao), ev(c%nao,c%nao)
    complex(dp) :: cc(c%nao,nb), nmat(c%nao,c%nao,2), dao(c%nao,c%nao,2), ddao(c%nao,c%nao,3,3,2)
    complex(dp), allocatable :: beta(:,:)
    real(dp) :: e_(c%nao), q(3,3), dq(3,3,3,2), clab(c%nao,c%nch), fcol(3,c%nao), fa(3,nat), ft(3), cond
    integer :: ik, info, p, k, sp, i, panc(3), ish, m
    allocate(beta(ldq,bt%nkb))
    call frame_of(c,tau,q,dq)
    do sp=1,2
      call ofe_dao(c%nao,c%nch,c%nshell,c%shell_l(1:c%nshell),c%shell_start(1:c%nshell),c%cloc(1:c%nao,1:c%nch), &
                   c%dch(1:c%nch,1:c%nch,sp),c%mode,q,dq,c%use_x,clab,dao(:,:,sp),ddao(:,:,:,:,sp))
    enddo
    nmat=0; fphi=0; fphid=0; faug=0; ffr=0; ftr=0
    do ik=1,nk
      sp=ik
      call build_phi(c,ik,tau,phi); call build_beta(ik,tau,beta)
      call ofe_group_build(npw(ik),ldq,c%nao,phi,chi,t,e_,ev,u,ldq,beta,bt,noop,info,cond)
      if(info/=0) stop 'group metric'
      call ofe_project(npw(ik),ldq,c%nao,nb,ldq,u,psi(:,:,ik),cc,noop)
      call ofe_nmat_acc(c%nao,nb,cc,w(:,ik),nmat(:,:,sp))
      if(want) then
        call ofe_force_k(npw(ik),ldq,c%nao,nb,qv(:,1:npw(ik),ik),phi,chi,t,e_,ev,u,ldq,psi(:,:,ik),w(:,ik),cc, &
                         dao(:,:,sp),ldq,beta,bt,fcol,fa,ft,noop)
        do ish=1,c%nshell
          do m=1,ofe_nlm(c%shell_l(ish))
            i=c%shell_start(ish)+m-1
            if(c%center_frozen) then
              fphid(:,c%shell_atom(ish))=fphid(:,c%shell_atom(ish))+fcol(:,i)
            else
              fphi(:,c%shell_atom(ish))=fphi(:,c%shell_atom(ish))+fcol(:,i)
            endif
          enddo
        enddo
        faug=faug+fa; ftr=ftr+ft
      endif
    enddo
    e=0
    do sp=1,2
      e=e+real(sum(dao(:,:,sp)*transpose(nmat(:,:,sp))),dp)
    enddo
    if(want .and. c%mode==2) then
      panc=[c%anc(2),c%anc(3),c%anc(1)]
      do sp=1,2
        do p=1,3
          if(p==2 .and. .not.c%use_x) cycle
          do k=1,3
            ffr(k,panc(p))=ffr(k,panc(p))-real(sum(nmat(:,:,sp)*transpose(ddao(:,:,k,p,sp))),dp)
          enddo
        enddo
      enddo
    endif
    if(present(nch_out)) then
      do sp=1,2
        nch_out(:,:,sp)=matmul(transpose(cmplx(clab,0._dp,dp)),matmul(nmat(:,:,sp),cmplx(clab,0._dp,dp)))
      enddo
    endif
  end subroutine energy_force

  real(dp) function energy(c,tau)
    type(case_t), intent(in) :: c
    real(dp), intent(in) :: tau(3,nat)
    real(dp) :: f1(3,nat), f2(3,nat), f3(3,nat), f4(3,nat), f5(3)
    call energy_force(c,tau,.false.,energy,f1,f2,f3,f4,f5)
  end function energy

  subroutine check_case(c)
    type(case_t), intent(in) :: c
    real(dp) :: e0, fphi(3,nat), fphid(3,nat), faug(3,nat), ffr(3,nat), ftr(3), fan(3,nat), ffd(3,nat)
    real(dp) :: h, err, scale, tp(3,nat), ep, em, dedv, delta, ed(2)
    complex(dp) :: nch(c%nch,c%nch,2), bmat(c%nch,c%nch)
    type(case_t) :: cp
    integer :: ia, k, i, j
    call energy_force(c,tau0,.true.,e0,fphi,fphid,faug,ffr,ftr,nch)
    fan=fphi+faug+ffr
    h=1.e-5_dp; err=0
    do ia=1,nat
      do k=1,3
        tp=tau0; tp(k,ia)=tp(k,ia)+h; ep=energy(c,tp)
        tp=tau0; tp(k,ia)=tp(k,ia)-h; em=energy(c,tp)
        ffd(k,ia)=-(ep-em)/(2*h)
        err=max(err,abs(ffd(k,ia)-fan(k,ia)))
      enddo
    enddo
    scale=max(1._dp,maxval(abs(fan)))
    write(*,'(a,1x,a,es12.4,a,es12.4)') 'CASE',trim(c%name),e0,'  max|F_expl| ',maxval(abs(fan))
    call report(trim(c%name)//'_explicit_force_vs_fd_all_atoms',err/scale,2.e-8_dp)
    call report(trim(c%name)//'_frame_sum_zero',maxval(abs(sum(ffr,dim=2))),1.e-12_dp)
    call report(trim(c%name)//'_sumrule_proj_aug_eq_commutator', &
                maxval(abs(sum(fphi+fphid+faug,dim=2)-ftr))/scale,1.e-11_dp)
    ! unrelated atom 6 (no beta, not in group, not anchor) must carry no explicit force
    call report(trim(c%name)//'_unrelated_atom_zero',maxval(abs(fan(:,6))),0._dp)
    ! atom 4: PAW, never in a group or frame -> aug only
    call report(trim(c%name)//'_paw_only_atom_aug_only',maxval(abs(fphi(:,4)))+maxval(abs(ffr(:,4))),0._dp)
    if(maxval(abs(faug(:,4)))<=0._dp) call report(trim(c%name)//'_paw_only_atom_has_aug',1._dp,0._dp)
    if(c%center_frozen) call report(trim(c%name)//'_frozen_centre_phi_zero',maxval(abs(fphi)),0._dp)
    ! HF linearity / dE/dV = Re Tr(B N): random Hermitian B on spin 2
    bmat=0
    do i=1,c%nch
      do j=i,c%nch
        bmat(i,j)=cmplx(0.3_dp*i-0.2_dp*j,0._dp,dp)
        if(i/=j) bmat(i,j)=bmat(i,j)+cmplx(0._dp,0.1_dp*(j-i),dp)
        bmat(j,i)=conjg(bmat(i,j))
      enddo
    enddo
    delta=1.e-3_dp
    cp=c; cp%dch(1:c%nch,1:c%nch,2)=c%dch(1:c%nch,1:c%nch,2)+delta*bmat; ed(1)=energy(cp,tau0)
    cp=c; cp%dch(1:c%nch,1:c%nch,2)=c%dch(1:c%nch,1:c%nch,2)-delta*bmat; ed(2)=energy(cp,tau0)
    dedv=real(sum(bmat*transpose(nch(:,:,2))),dp)
    call report(trim(c%name)//'_hf_dEdV_eq_TrBN',abs((ed(1)-ed(2))/(2*delta)-dedv),1.e-11_dp)
  end subroutine check_case

  subroutine set_diag_local(c,v)
    type(case_t), intent(inout) :: c
    real(dp), intent(in) :: v(:)
    integer :: i
    c%dch=0
    do i=1,size(v)
      c%dch(i,i,1)=v(i); c%dch(i,i,2)=v(i)
    enddo
  end subroutine set_diag_local

  subroutine make_cases()
    real(dp) :: x(mao,mao), m(8,8)
    integer :: i, j
    ! 1 RU_D: atom 1 d shell, C = I5, moving frame origin 1, z_from 2 (image -a3), x_from 5
    cs(1)%name='RU_D'; cs(1)%nao=5; cs(1)%nch=5; cs(1)%nshell=1
    cs(1)%shell_l(1)=2; cs(1)%shell_start(1)=1; cs(1)%shell_atom(1)=1; cs(1)%shell_alpha(1)=0.9_dp
    do i=1,5
      cs(1)%cloc(i,i)=1
    enddo
    call rand_mat(m(1:5,1:5))
    cs(1)%dch(1:5,1:5,1)=cmplx(0.1_dp*(m(1:5,1:5)+transpose(m(1:5,1:5))),0._dp,dp)
    cs(1)%dch(:,:,2)=cs(1)%dch(:,:,1)
    cs(1)%anc=[1,2,5]; cs(1)%anc_img(:,2)=[0,0,-1]
    ! 2 O_P_BOND: atom 2 p shell (image -a3), frame origin = partner 1, z_from = 2, helper x axis
    cs(2)%name='O_P_BOND_helper'; cs(2)%nao=3; cs(2)%nch=3; cs(2)%nshell=1
    cs(2)%shell_l(1)=1; cs(2)%shell_start(1)=1; cs(2)%shell_atom(1)=2; cs(2)%shell_img(:,1)=[0,0,-1]
    cs(2)%shell_alpha(1)=0.8_dp
    do i=1,3
      cs(2)%cloc(i,i)=1
    enddo
    call set_diag_local(cs(2),[0.04_dp,0.04_dp,-0.03_dp])
    cs(2)%anc=[1,2,0]; cs(2)%anc_img(:,2)=[0,0,-1]; cs(2)%use_x=.false.; cs(2)%helper=[1._dp,0._dp,0._dp]
    ! 3 RU_O_PI: joint Ru d (atom 1) + O p (atom 2, image -a3); channels (d_xz,d_yz,p_x,p_y); complex D
    cs(3)%name='RU_O_PI_joint_complexD'; cs(3)%nao=8; cs(3)%nch=4; cs(3)%nshell=2
    cs(3)%shell_l(1:2)=[2,1]; cs(3)%shell_start(1:2)=[1,6]; cs(3)%shell_atom(1:2)=[1,2]
    cs(3)%shell_img(:,2)=[0,0,-1]; cs(3)%shell_alpha(1:2)=[0.9_dp,0.8_dp]
    cs(3)%cloc(2,1)=1; cs(3)%cloc(3,2)=1; cs(3)%cloc(6,3)=1; cs(3)%cloc(7,4)=1
    cs(3)%dch(1,1,1)=0.02_dp; cs(3)%dch(3,3,1)=-0.03_dp; cs(3)%dch(1,3,1)=cmplx(0.05_dp,0.02_dp,dp)
    cs(3)%dch(3,1,1)=conjg(cs(3)%dch(1,3,1)); cs(3)%dch(2,4,1)=-0.04_dp; cs(3)%dch(4,2,1)=-0.04_dp
    cs(3)%dch(:,:,2)=cs(3)%dch(:,:,1)
    cs(3)%anc=[1,2,3]; cs(3)%anc_img(:,2)=[0,0,-1]
    ! 4 IAO_CHANNEL: atom 1 s+d, atom 2 p (image), atom 3 p; random C (s row zero = Lowdin-only shell);
    !   spin-resolved D (shared C)
    cs(4)%name='IAO_joint_spinresolved'; cs(4)%nao=12; cs(4)%nch=4; cs(4)%nshell=4
    cs(4)%shell_l(1:4)=[0,2,1,1]; cs(4)%shell_start(1:4)=[1,2,7,10]; cs(4)%shell_atom(1:4)=[1,1,2,3]
    cs(4)%shell_img(:,3)=[0,0,-1]; cs(4)%shell_alpha(1:4)=[0.55_dp,0.9_dp,0.8_dp,0.8_dp]
    call rand_mat(x(1:12,1:4)); x(1,1:4)=0; call orthonormalize(x(1:12,1:4)); cs(4)%cloc(1:12,1:4)=x(1:12,1:4)
    call rand_mat(m(1:4,1:4))
    cs(4)%dch(1:4,1:4,1)=cmplx(0.05_dp*(m(1:4,1:4)+transpose(m(1:4,1:4))),0._dp,dp)
    call rand_mat(m(1:4,1:4))
    cs(4)%dch(1:4,1:4,2)=cmplx(0.05_dp*(m(1:4,1:4)+transpose(m(1:4,1:4))),0._dp,dp)
    cs(4)%anc=[1,2,5]; cs(4)%anc_img(:,2)=[0,0,-1]
    ! 5 frozen centre (RU_D geometry, Phi fixed at tau0 positions)
    cs(5)=cs(1); cs(5)%name='RU_D_frozen_centre'; cs(5)%center_frozen=.true.
    cs(5)%cref(:,1)=tau0(:,1)
    ! 6 IAO joint, frozen frame from tau0 / global variant alternate: use frozen here, global below
    cs(6)=cs(4); cs(6)%name='IAO_frozen_frame'; cs(6)%mode=1
    call frame_q0(cs(4),cs(6)%qref)
  end subroutine make_cases

  subroutine frame_q0(c,q)
    type(case_t), intent(in) :: c
    real(dp), intent(out) :: q(3,3)
    real(dp) :: dq(3,3,3,2)
    call frame_of(c,tau0,q,dq)
  end subroutine frame_q0

  subroutine check_stack_and_hermitian()
    ! two sites (RU_D + IAO joint) at k1: stacked apply == sum of per-site W D W^H psi; H Hermitian
    type(case_t) :: c
    integer, parameter :: ns=2
    integer :: ik, is, info, nc(ns), off, i, j
    complex(dp), allocatable :: beta(:,:), wall(:,:), dall(:,:), hp1(:,:), hp2(:,:), phi(:,:), chi(:,:), u(:,:)
    complex(dp), allocatable :: t(:,:), ev(:,:), ws(:,:), hm(:,:)
    real(dp), allocatable :: e_(:), clab(:,:)
    complex(dp) :: dao(16,16), ddao(16,16,3,3)
    real(dp) :: q(3,3), dq(3,3,3,2), cond, err
    ik=1
    allocate(beta(ldq,bt%nkb))
    call build_beta(ik,tau0,beta)
    nc=[cs(1)%nch,cs(4)%nch]
    allocate(wall(ldq,sum(nc)),dall(sum(nc),sum(nc)),hp1(ldq,nb),hp2(ldq,nb))
    wall=0; dall=0; hp1=0; hp2=0; off=0
    do is=1,ns
      if(is==1) c=cs(1)
      if(is==2) c=cs(4)
      allocate(phi(ldq,c%nao),chi(ldq,c%nao),u(ldq,c%nao),t(c%nao,c%nao),ev(c%nao,c%nao),e_(c%nao), &
               clab(c%nao,c%nch),ws(ldq,c%nch))
      call build_phi(c,ik,tau0,phi)
      call ofe_group_build(npw(ik),ldq,c%nao,phi,chi,t,e_,ev,u,ldq,beta,bt,noop,info,cond)
      call frame_of(c,tau0,q,dq)
      call ofe_dao(c%nao,c%nch,c%nshell,c%shell_l(1:c%nshell),c%shell_start(1:c%nshell),c%cloc(1:c%nao,1:c%nch), &
                   c%dch(1:c%nch,1:c%nch,1),c%mode,q,dq,c%use_x,clab,dao(1:c%nao,1:c%nao),ddao(1:c%nao,1:c%nao,:,:))
      ws=matmul(u,cmplx(clab,0._dp,dp))
      wall(:,off+1:off+c%nch)=ws
      dall(off+1:off+c%nch,off+1:off+c%nch)=c%dch(1:c%nch,1:c%nch,1)
      hp1=hp1+matmul(ws,matmul(c%dch(1:c%nch,1:c%nch,1),matmul(transpose(conjg(ws)),psi(:,:,ik))))
      off=off+c%nch
      deallocate(phi,chi,u,t,ev,e_,clab,ws)
    enddo
    call ofe_apply_stack(npw(ik),ldq,nb,sum(nc),wall,dall,psi(:,:,ik),hp2,noop)
    call report('stacked_apply_eq_sum_of_sites',maxval(abs(hp1-hp2)),1.e-13_dp)
    allocate(hm(nb,nb))
    hm=matmul(transpose(conjg(psi(:,:,ik))),hp2)
    err=0
    do i=1,nb
      do j=1,nb
        err=max(err,abs(hm(i,j)-conjg(hm(j,i))))
      enddo
    enddo
    call report('h_ext_hermitian_on_psi',err,1.e-13_dp)
  end subroutine check_stack_and_hermitian

  subroutine check_axial()
    real(dp) :: e1, e2
    e1=ofe_axial_err(cs(2)%nao,cs(2)%nch,1,cs(2)%shell_l(1:1),cs(2)%shell_start(1:1),cs(2)%cloc(1:3,1:3), &
                     cs(2)%dch(1:3,1:3,1))
    e2=ofe_axial_err(cs(1)%nao,cs(1)%nch,1,cs(1)%shell_l(1:1),cs(1)%shell_start(1:1),cs(1)%cloc(1:5,1:5), &
                     cs(1)%dch(1:5,1:5,1))
    call report('axial_check_sigma_pi_isotropic_is_zero',e1,1.e-14_dp)
    call report('axial_check_detects_general_d_field',merge(0._dp,1._dp,e2>1.e-3_dp),0._dp)
  end subroutine check_axial

  subroutine check_inv_sqrt()
    integer, parameter :: n=9
    real(dp) :: a(n,n), b(n,n), e(n), ep(n), err, hh
    complex(dp) :: g(n,n), x(n,n), t(n,n), ev(n,n), tp(n,n), tm(n,n), evp(n,n), la(n,n), lb(n,n), y(n,n)
    integer :: info, i
    call rand_mat(a); call rand_mat(b); x=cmplx(a,b,dp)
    g=matmul(x,transpose(conjg(x)))/n
    do i=1,n
      g(i,i)=g(i,i)+0.3_dp
    enddo
    call ofe_inv_sqrt(n,g,t,e,ev,info)
    call rand_mat(a); call rand_mat(b); x=cmplx(a,b,dp); x=x+transpose(conjg(x))
    call rand_mat(a); call rand_mat(b); y=cmplx(a,b,dp); y=y+transpose(conjg(y))
    hh=1.e-6_dp
    call ofe_inv_sqrt(n,g+hh*x,tp,ep,evp,info); call ofe_inv_sqrt(n,g-hh*x,tm,ep,evp,info)
    call ofe_frechet(n,e,ev,x,la); call ofe_frechet(n,e,ev,y,lb)
    err=maxval(abs((tp-tm)/(2*hh)-la))
    err=max(err,abs(sum(y*transpose(la))-sum(lb*transpose(x))))
    err=max(err,maxval(abs(matmul(t,matmul(g,t))-eye(n))))
    call report('inv_sqrt_frechet_fd_adjoint',err,1.e-7_dp)
  end subroutine check_inv_sqrt

  subroutine random_rotation(q)
    real(dp), intent(out) :: q(3,3)
    call rand_mat(q); call orthonormalize(q)
    if(det3(q)<0) q(:,3)=-q(:,3)
  end subroutine random_rotation

  subroutine check_rot()
    real(dp) :: q1(3,3), q2(3,3), r1(5,5), r2(5,5), r12(5,5), err
    integer :: l, n
    call random_rotation(q1); call random_rotation(q2); err=0
    do l=0,2
      n=ofe_nlm(l)
      call ofe_rot_block(l,q1,r1(1:n,1:n)); call ofe_rot_block(l,q2,r2(1:n,1:n))
      call ofe_rot_block(l,matmul(q1,q2),r12(1:n,1:n))
      err=max(err,maxval(abs(matmul(r1(1:n,1:n),r2(1:n,1:n))-r12(1:n,1:n))))
      err=max(err,maxval(abs(matmul(transpose(r1(1:n,1:n)),r1(1:n,1:n))-real(eye(n),dp))))
    enddo
    call report('rotation_blocks_homomorphism_orthogonal',err,1.e-12_dp)
  end subroutine check_rot

  subroutine check_frame()
    real(dp) :: a(3), b(3), q(3,3), dq(3,3,3,2), qp(3,3), qm(3,3), dd(3,3,3,2), d(3)
    real(dp) :: rp(5,5), rm(5,5), dr(5,5), hh, err
    integer :: info, k, p, l, n
    call random_number(a); call random_number(b); a=4*a-2; b=4*b-2
    call ofe_frame(a,b,q,dq,info); hh=1.e-6_dp; err=0
    do p=1,2
      do k=1,3
        d=0; d(k)=hh
        if(p==1) then
          call ofe_frame(a+d,b,qp,dd,info); call ofe_frame(a-d,b,qm,dd,info)
        else
          call ofe_frame(a,b+d,qp,dd,info); call ofe_frame(a,b-d,qm,dd,info)
        endif
        err=max(err,maxval(abs((qp-qm)/(2*hh)-dq(:,:,k,p))))
        do l=1,2
          n=ofe_nlm(l)
          call ofe_rot_block(l,qp,rp(1:n,1:n)); call ofe_rot_block(l,qm,rm(1:n,1:n))
          call ofe_drot_block(l,q,dq(:,:,k,p),dr(1:n,1:n))
          err=max(err,maxval(abs((rp(1:n,1:n)-rm(1:n,1:n))/(2*hh)-dr(1:n,1:n))))
        enddo
      enddo
    enddo
    call report('frame_and_drot_fd',err,1.e-8_dp)
    call ofe_frame(a,2*a,q,dq,info)
    call report('collinear_anchor_detected',merge(0._dp,1._dp,info==2),0._dp)
  end subroutine check_frame

  subroutine check_ylm()
    real(dp) :: q(3,3), v(3), y0(5), y1(5), r(5,5), err
    integer :: l, n, i
    call random_rotation(q); err=0
    do i=1,10
      call random_number(v); v=2*v-1
      do l=0,2
        n=ofe_nlm(l)
        call ofe_rot_block(l,q,r(1:n,1:n))
        call ofe_ylm_cart(l,matmul(transpose(q),v),y1(1:n)); call ofe_ylm_cart(l,v,y0(1:n))
        err=max(err,maxval(abs(y1(1:n)-matmul(y0(1:n),r(1:n,1:n)))))
      enddo
    enddo
    call report('ylm_cart_rotation_relation',err,1.e-12_dp)
  end subroutine check_ylm
end program test_ofe_math
