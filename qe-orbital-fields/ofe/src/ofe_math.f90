! Orbital Field Engine (OFE_V1): QE-independent algebra kernels. GPL-2.0-or-later.
! Merges the vnext kernels of the four review lines (2026-10-25):
!   ic_math (IAO_CHANNEL_V1: multi-atom groups, frozen channel C, rotation blocks l<=2),
!   rd_kernels (Ru d vnext: batched beta projections, translation sum rule, stacked apply),
!   bf_math / top_pi_kernels (same reverse-mode formulas, single-centre / joint d+p special cases).
! QE data (vkb, qq_at, ofsbeta, nh) and the plane-wave reduction (mp_sum over intra_bgrp_comm) are
! passed in, so the standalone test (tests/fortran/test_ofe_math.f90) runs the production code.
!
! Per k and per site (group of atoms x shells):
!   Phi = group AOs, coefficient (-i)^l Y_lm(q) chi_l(|q|) exp(-i q.(tau+A img)), positive Cartesian
!   chi = S Phi, G = Phi^H chi, T = G^(-1/2), U = chi T (U^H S^-1 U = 1), W = U C_lab,
!   c = U^H psi, N_ao = sum_b w_b c_b c_b^H, D_ao = C_lab D C_lab^T, E = Tr(D_ao N_ao) = Tr(D N_ch).
! Forces are -dE/dtau at fixed psi (Hellmann-Feynman part owned by the engine).
module ofe_math
  use iso_fortran_env, only: real64
  implicit none
  private
  integer, parameter, public :: ofe_dp = real64
  complex(ofe_dp), parameter :: one=(1._ofe_dp,0._ofe_dp), zero=(0._ofe_dp,0._ofe_dp)
  abstract interface
    subroutine ofe_reduce_iface(a)
      import :: ofe_dp
      complex(ofe_dp), intent(inout) :: a(:,:)
    end subroutine ofe_reduce_iface
  end interface
  ! beta-projector bookkeeping (QE: nkb, nhm, ofsbeta, nh(ityp), upf%tvanp, qq_at)
  type, public :: ofe_beta_t
    integer :: nat=0, nhm=0, nkb=0
    integer, allocatable :: ofs(:), nh(:)
    logical, allocatable :: us(:)
    real(ofe_dp), allocatable :: qq(:,:,:)
  end type ofe_beta_t
  public :: ofe_reduce_iface
  public :: ofe_nlm, ofe_inv_sqrt, ofe_frechet, ofe_d_basis, ofe_rot_block, ofe_drot_block, ofe_frame
  public :: ofe_rz, ofe_ylm_cart, ofe_cart_from_qe_ylm, ofe_s_apply, ofe_group_build, ofe_project
  public :: ofe_nmat_acc, ofe_force_k, ofe_dao, ofe_axial_err, ofe_apply_stack
contains
  pure integer function ofe_nlm(l)
    integer, intent(in) :: l
    ofe_nlm=2*l+1
  end function ofe_nlm

  ! ------------------------------------------------------------------ matrix functions
  subroutine ofe_inv_sqrt(n,g,t,eval,evec,info)
    ! T = G^(-1/2); info=-100 non-Hermitian, -101 ill-conditioned (cond > 1e10 or eig <= 1e-12)
    integer, intent(in) :: n
    complex(ofe_dp), intent(in) :: g(n,n)
    complex(ofe_dp), intent(out) :: t(n,n), evec(n,n)
    real(ofe_dp), intent(out) :: eval(n)
    integer, intent(out) :: info
    complex(ofe_dp), allocatable :: work(:), v(:,:)
    real(ofe_dp), allocatable :: rwork(:)
    real(ofe_dp) :: err
    integer :: j, lwork
    info=0; t=zero; evec=zero; eval=0
    err=maxval(abs(g-transpose(conjg(g))))
    if(err>1.e-10_ofe_dp*max(1._ofe_dp,maxval(abs(g)))) then
      info=-100; return
    endif
    evec=0.5_ofe_dp*(g+transpose(conjg(g)))
    lwork=max(1,64*n)
    allocate(work(lwork),rwork(max(1,3*n-2)),v(n,n))
    call zheev('V','U',n,evec,n,eval,work,lwork,rwork,info)
    if(info/=0) return
    if(minval(eval)<=1.e-12_ofe_dp .or. maxval(eval)/minval(eval)>1.e10_ofe_dp) then
      info=-101; return
    endif
    do j=1,n
      v(:,j)=evec(:,j)/sqrt(eval(j))
    enddo
    t=matmul(v,transpose(conjg(evec)))
    deallocate(work,rwork,v)
  end subroutine ofe_inv_sqrt

  subroutine ofe_frechet(n,eval,evec,a,la)
    ! Frechet derivative of G^(-1/2) at G (eigenpairs eval, evec) along Hermitian a;
    ! self-adjoint under the trace pairing: Tr(B L(A)) = Tr(L(B) A).
    integer, intent(in) :: n
    real(ofe_dp), intent(in) :: eval(n)
    complex(ofe_dp), intent(in) :: evec(n,n), a(n,n)
    complex(ofe_dp), intent(out) :: la(n,n)
    complex(ofe_dp) :: v(n,n)
    integer :: i, j
    v=matmul(transpose(conjg(evec)),matmul(a,evec))
    do j=1,n
      do i=1,n
        v(i,j)=-v(i,j)/(sqrt(eval(i))*sqrt(eval(j))*(sqrt(eval(i))+sqrt(eval(j))))
      enddo
    enddo
    la=matmul(evec,matmul(v,transpose(conjg(evec))))
  end subroutine ofe_frechet

  ! ------------------------------------------------------------------ real harmonics, rotations
  subroutine ofe_d_basis(b)
    ! Frobenius-orthonormal d tensors, order z2, xz, yz, x2-y2, xy (as rd_basis / ic_d_basis)
    real(ofe_dp), intent(out) :: b(3,3,5)
    b=0
    b(1,1,1)=-1/sqrt(6._ofe_dp); b(2,2,1)=b(1,1,1); b(3,3,1)=2/sqrt(6._ofe_dp)
    b(1,3,2)=1/sqrt(2._ofe_dp); b(3,1,2)=b(1,3,2)
    b(2,3,3)=1/sqrt(2._ofe_dp); b(3,2,3)=b(2,3,3)
    b(1,1,4)=1/sqrt(2._ofe_dp); b(2,2,4)=-b(1,1,4)
    b(1,2,5)=1/sqrt(2._ofe_dp); b(2,1,5)=b(1,2,5)
  end subroutine ofe_d_basis

  subroutine ofe_rot_block(l,q,r)
    ! local orbital y'_j(r) = y_j(Q^T r) = sum_i R_ij y_i(r); Q columns = local x,y,z (lab coords)
    integer, intent(in) :: l
    real(ofe_dp), intent(in) :: q(3,3)
    real(ofe_dp), intent(out) :: r(:,:)
    real(ofe_dp) :: b(3,3,5), t(3,3)
    integer :: i, j
    r=0
    select case(l)
    case(0)
      r(1,1)=1
    case(1)
      r(1:3,1:3)=q
    case(2)
      call ofe_d_basis(b)
      do j=1,5
        t=matmul(q,matmul(b(:,:,j),transpose(q)))
        do i=1,5
          r(i,j)=sum(b(:,:,i)*t)
        enddo
      enddo
    end select
  end subroutine ofe_rot_block

  subroutine ofe_drot_block(l,q,dq,dr)
    integer, intent(in) :: l
    real(ofe_dp), intent(in) :: q(3,3), dq(3,3)
    real(ofe_dp), intent(out) :: dr(:,:)
    real(ofe_dp) :: b(3,3,5), t(3,3)
    integer :: i, j
    dr=0
    select case(l)
    case(1)
      dr(1:3,1:3)=dq
    case(2)
      call ofe_d_basis(b)
      do j=1,5
        t=matmul(dq,matmul(b(:,:,j),transpose(q)))+matmul(q,matmul(b(:,:,j),transpose(dq)))
        do i=1,5
          dr(i,j)=sum(b(:,:,i)*t)
        enddo
      enddo
    end select
  end subroutine ofe_drot_block

  function cross(a,b) result(c)
    real(ofe_dp), intent(in) :: a(3), b(3)
    real(ofe_dp) :: c(3)
    c=[a(2)*b(3)-a(3)*b(2),a(3)*b(1)-a(1)*b(3),a(1)*b(2)-a(2)*b(1)]
  end function cross

  subroutine ofe_frame(a,b,q,dq,info)
    ! z = a/|a|, x = (b - z(z.b))/|..|, y = z x x;  a = z_from - origin, b = x_from - origin
    ! (or a constant helper vector).  dq(:,:,k,p): derivative w.r.t. a_k (p=1), b_k (p=2).
    real(ofe_dp), intent(in) :: a(3), b(3)
    real(ofe_dp), intent(out) :: q(3,3), dq(3,3,3,2)
    integer, intent(out) :: info
    real(ofe_dp) :: z(3), x(3), w(3), na, nw, da(3), db(3), dz(3), dw(3), dx(3)
    integer :: k, p
    info=0; q=0; dq=0
    na=sqrt(sum(a*a))
    if(na<1.e-10_ofe_dp) then
      info=1; return
    endif
    z=a/na; w=b-z*dot_product(z,b); nw=sqrt(sum(w*w))
    if(nw<1.e-10_ofe_dp*max(1._ofe_dp,sqrt(sum(b*b)))) then
      info=2; return
    endif
    x=w/nw
    q(:,1)=x; q(:,2)=cross(z,x); q(:,3)=z
    do p=1,2
      do k=1,3
        da=0; db=0
        if(p==1) da(k)=1
        if(p==2) db(k)=1
        dz=(da-z*dot_product(z,da))/na
        dw=db-dz*dot_product(z,b)-z*(dot_product(dz,b)+dot_product(z,db))
        dx=(dw-x*dot_product(x,dw))/nw
        dq(:,1,k,p)=dx; dq(:,2,k,p)=cross(dz,x)+cross(z,dx); dq(:,3,k,p)=dz
      enddo
    enddo
  end subroutine ofe_frame

  function ofe_rz(theta) result(r)
    ! rotation about the local z axis (columns = rotated x, y, z)
    real(ofe_dp), intent(in) :: theta
    real(ofe_dp) :: r(3,3)
    r=0
    r(1,1)=cos(theta); r(2,1)=sin(theta); r(1,2)=-sin(theta); r(2,2)=cos(theta); r(3,3)=1
  end function ofe_rz

  subroutine ofe_ylm_cart(l,v,y)
    ! analytic normalised real harmonics, positive Cartesian (no Condon-Shortley sign)
    integer, intent(in) :: l
    real(ofe_dp), intent(in) :: v(3)
    real(ofe_dp), intent(out) :: y(:)
    real(ofe_dp) :: u(3), b(3,3,5), pi
    integer :: j
    pi=acos(-1._ofe_dp)
    u=v/max(sqrt(sum(v*v)),tiny(1._ofe_dp))
    select case(l)
    case(0)
      y(1)=1/sqrt(4*pi)
    case(1)
      y(1:3)=sqrt(3/(4*pi))*u
    case(2)
      call ofe_d_basis(b)
      do j=1,5
        y(j)=sqrt(15/(8*pi))*dot_product(u,matmul(b(:,:,j),u))
      enddo
    end select
  end subroutine ofe_ylm_cart

  subroutine ofe_cart_from_qe_ylm(l,ylm,y)
    ! QE ylmr2 order/sign -> positive Cartesian: l=1 (+z,-x,-y) -> (x,y,z);
    ! l=2 (+z2,-xz,-yz,+x2-y2,+xy) -> (z2,xz,yz,x2-y2,xy).  ylm = one row ylm(1:9).
    integer, intent(in) :: l
    real(ofe_dp), intent(in) :: ylm(9)
    real(ofe_dp), intent(out) :: y(:)
    select case(l)
    case(0)
      y(1)=ylm(1)
    case(1)
      y(1:3)=[-ylm(3),-ylm(4),ylm(2)]
    case(2)
      y(1:5)=[ylm(5),-ylm(6),-ylm(7),ylm(8),ylm(9)]
    end select
  end subroutine ofe_cart_from_qe_ylm

  ! ------------------------------------------------------------------ PAW overlap and group basis
  subroutine ofe_s_apply(n,lda,m,a,sa,ldb,beta,bt,reduce)
    ! sa = S a = a + sum_J beta_J q_J beta_J^H a (rows 1:n; rows n+1:lda zeroed)
    integer, intent(in) :: n, lda, m, ldb
    complex(ofe_dp), intent(in) :: a(lda,m), beta(ldb,*)
    complex(ofe_dp), intent(out) :: sa(lda,m)
    type(ofe_beta_t), intent(in) :: bt
    procedure(ofe_reduce_iface) :: reduce
    complex(ofe_dp), allocatable :: bp(:,:), qb(:,:)
    integer :: j, lo, hi, nh
    sa=zero
    sa(1:n,:)=a(1:n,:)
    if(bt%nkb==0 .or. m==0) return
    if(.not.any(bt%us)) return
    allocate(bp(bt%nkb,m),qb(bt%nkb,m))
    call zgemm('C','N',bt%nkb,m,n,one,beta,ldb,a,lda,zero,bp,bt%nkb)
    call reduce(bp)
    qb=zero
    do j=1,bt%nat
      nh=bt%nh(j)
      if(.not.bt%us(j) .or. nh==0) cycle
      lo=bt%ofs(j)+1; hi=bt%ofs(j)+nh
      qb(lo:hi,:)=matmul(cmplx(bt%qq(1:nh,1:nh,j),0._ofe_dp,ofe_dp),bp(lo:hi,:))
    enddo
    call zgemm('N','N',n,m,bt%nkb,one,beta,ldb,qb,bt%nkb,one,sa,lda)
    deallocate(bp,qb)
  end subroutine ofe_s_apply

  subroutine ofe_group_build(n,lda,nao,phi,chi,t,eval,evec,u,ldb,beta,bt,reduce,info,cond)
    ! chi = S Phi, G = Phi^H chi, T = G^(-1/2), U = chi T; info /= 0 on failure
    ! (-100 non-Hermitian, -101 ill-conditioned, -102 T G T /= 1 beyond 1e-9).
    integer, intent(in) :: n, lda, nao, ldb
    complex(ofe_dp), intent(in) :: phi(lda,nao), beta(ldb,*)
    complex(ofe_dp), intent(out) :: chi(lda,nao), t(nao,nao), evec(nao,nao), u(lda,nao)
    real(ofe_dp), intent(out) :: eval(nao), cond
    type(ofe_beta_t), intent(in) :: bt
    procedure(ofe_reduce_iface) :: reduce
    integer, intent(out) :: info
    complex(ofe_dp) :: g(nao,nao), ident(nao,nao)
    integer :: i
    call ofe_s_apply(n,lda,nao,phi,chi,ldb,beta,bt,reduce)
    call zgemm('C','N',nao,nao,n,one,phi,lda,chi,lda,zero,g,nao)
    call reduce(g)
    call ofe_inv_sqrt(nao,g,t,eval,evec,info)
    cond=0
    if(info/=0) return
    cond=maxval(eval)/minval(eval)
    ident=matmul(t,matmul(g,t))
    do i=1,nao
      ident(i,i)=ident(i,i)-1
    enddo
    if(maxval(abs(ident))>1.e-9_ofe_dp) then
      info=-102; return
    endif
    u=zero
    call zgemm('N','N',n,nao,nao,one,chi,lda,t,nao,zero,u,lda)
  end subroutine ofe_group_build

  subroutine ofe_project(n,lda,ncol,nb,ldpsi,u,psi,c,reduce)
    ! c(ncol,nb) = U^H psi for all stacked AO columns of all sites: one ZGEMM, one reduction
    integer, intent(in) :: n, lda, ncol, nb, ldpsi
    complex(ofe_dp), intent(in) :: u(lda,ncol), psi(ldpsi,nb)
    complex(ofe_dp), intent(out) :: c(ncol,nb)
    procedure(ofe_reduce_iface) :: reduce
    call zgemm('C','N',ncol,nb,n,one,u,lda,psi,ldpsi,zero,c,ncol)
    call reduce(c)
  end subroutine ofe_project

  subroutine ofe_nmat_acc(nao,nb,c,w,nmat)
    ! nmat(i,j) += sum_b w_b c(i,b) conj(c(j,b))
    integer, intent(in) :: nao, nb
    complex(ofe_dp), intent(in) :: c(nao,nb)
    real(ofe_dp), intent(in) :: w(nb)
    complex(ofe_dp), intent(inout) :: nmat(nao,nao)
    complex(ofe_dp), allocatable :: cw(:,:)
    integer :: ib
    allocate(cw(nao,nb))
    do ib=1,nb
      cw(:,ib)=w(ib)*c(:,ib)
    enddo
    call zgemm('N','C',nao,nao,nb,one,cw,nao,c,nao,one,nmat,nao)
    deallocate(cw)
  end subroutine ofe_nmat_acc

  subroutine ofe_force_k(n,lda,nao,nb,q,phi,chi,t,eval,evec,u,ldpsi,psi,w,c,dao,ldb,beta,bt, &
                         fcol,faug,ftr,reduce)
    ! Reverse-mode explicit force of E = Tr(D_ao N_ao) at fixed psi, one k point, one site:
    !   Z = Psi f c^H D_ao, X = Z T, K = 2 L_{G^-1/2}(herm(chi^H Z)), gradPhi = S X + chi K
    !   fcol(:,a) = -2 Re <gradPhi_a | -i q Phi_a>      (centre of AO column a moves)
    !   faug(:,J) = -2 Re[Tr(c1 dbp_J) + Tr(c2 dbx_J)]  (S moves with PAW atom J), with
    !     c2 = (beta_J^H Phi)^H q_J, c1 = (beta_J^H X)^H q_J + K c2, dbp = beta^H(iq Phi), dbx = beta^H(iq X)
    !   ftr = -2 Re <Z, -i q U>  (translation identity: sum_a fcol + sum_J faug = ftr exactly)
    ! All beta projections of [Phi, X, iq Phi, iq X] in ONE ZGEMM and ONE reduction.
    integer, intent(in) :: n, lda, nao, nb, ldpsi, ldb
    real(ofe_dp), intent(in) :: q(3,n), eval(nao), w(nb)
    complex(ofe_dp), intent(in) :: phi(lda,nao), chi(lda,nao), t(nao,nao), evec(nao,nao), u(lda,nao)
    complex(ofe_dp), intent(in) :: psi(ldpsi,nb), c(nao,nb), dao(nao,nao), beta(ldb,*)
    type(ofe_beta_t), intent(in) :: bt
    real(ofe_dp), intent(out) :: fcol(3,nao), faug(3,bt%nat), ftr(3)
    procedure(ofe_reduce_iface) :: reduce
    complex(ofe_dp), allocatable :: coef(:,:), z(:,:), x(:,:), grad(:,:), tmp(:,:), proj(:,:), qb(:,:)
    complex(ofe_dp), allocatable :: aa(:,:), kk(:,:), c1(:,:), c2(:,:), qqc(:,:), buf(:,:)
    integer :: ib, ip, ia, j, nh, lo, hi, o1, o2, nk
    logical :: useaug
    allocate(coef(nb,nao),z(n,nao),x(n,nao),grad(n,nao),aa(nao,nao),kk(nao,nao),buf(3,nao+1))
    coef=matmul(transpose(conjg(c)),dao)
    do ib=1,nb
      coef(ib,:)=w(ib)*coef(ib,:)
    enddo
    call zgemm('N','N',n,nao,nb,one,psi,ldpsi,coef,nb,zero,z,n)        ! Z = rho U D_ao
    call zgemm('N','N',n,nao,nao,one,z,n,t,nao,zero,x,n)               ! X = Z T
    call zgemm('C','N',nao,nao,n,one,chi,lda,z,n,zero,aa,nao)
    call reduce(aa)
    aa=0.5_ofe_dp*(aa+transpose(conjg(aa)))
    call ofe_frechet(nao,eval,evec,aa,kk)
    kk=2*kk
    nk=bt%nkb
    useaug=.false.
    if(nk>0) useaug=any(bt%us)
    if(useaug) then
      allocate(tmp(n,8*nao),proj(nk,8*nao))
      tmp(:,1:nao)=phi(1:n,:)
      tmp(:,nao+1:2*nao)=x
      do ip=1,3
        do ia=1,nao
          tmp(:,(1+ip)*nao+ia)=cmplx(0._ofe_dp,q(ip,1:n),ofe_dp)*phi(1:n,ia)
          tmp(:,(4+ip)*nao+ia)=cmplx(0._ofe_dp,q(ip,1:n),ofe_dp)*x(:,ia)
        enddo
      enddo
      call zgemm('C','N',nk,8*nao,n,one,beta,ldb,tmp,n,zero,proj,nk)
      call reduce(proj)
      deallocate(tmp)
    endif
    grad=x                                                             ! S X = X + beta q beta^H X
    if(useaug) then
      allocate(qb(nk,nao)); qb=zero
      do j=1,bt%nat
        nh=bt%nh(j)
        if(.not.bt%us(j) .or. nh==0) cycle
        lo=bt%ofs(j)+1; hi=bt%ofs(j)+nh
        qb(lo:hi,:)=matmul(cmplx(bt%qq(1:nh,1:nh,j),0._ofe_dp,ofe_dp),proj(lo:hi,nao+1:2*nao))
      enddo
      call zgemm('N','N',n,nao,nk,one,beta,ldb,qb,nk,one,grad,n)
      deallocate(qb)
    endif
    call zgemm('N','N',n,nao,nao,one,chi,lda,kk,nao,one,grad,n)        ! + chi K
    buf=zero
    do ia=1,nao
      do ip=1,3
        buf(ip,ia)=cmplx(-2*real(sum(conjg(grad(:,ia))*cmplx(0._ofe_dp,-q(ip,1:n),ofe_dp)*phi(1:n,ia)), &
                   ofe_dp),0._ofe_dp,ofe_dp)
        buf(ip,nao+1)=buf(ip,nao+1)+cmplx(-2*real(sum(conjg(z(:,ia))*cmplx(0._ofe_dp,-q(ip,1:n),ofe_dp) &
                      *u(1:n,ia)),ofe_dp),0._ofe_dp,ofe_dp)
      enddo
    enddo
    call reduce(buf)
    fcol=real(buf(:,1:nao),ofe_dp)
    ftr=real(buf(:,nao+1),ofe_dp)
    faug=0
    if(useaug) then
      allocate(c1(nao,bt%nhm),c2(nao,bt%nhm),qqc(bt%nhm,bt%nhm))
      do j=1,bt%nat
        nh=bt%nh(j)
        if(.not.bt%us(j) .or. nh==0) cycle
        lo=bt%ofs(j)+1; hi=bt%ofs(j)+nh
        qqc(1:nh,1:nh)=cmplx(bt%qq(1:nh,1:nh,j),0._ofe_dp,ofe_dp)
        c2(:,1:nh)=matmul(transpose(conjg(proj(lo:hi,1:nao))),qqc(1:nh,1:nh))
        c1(:,1:nh)=matmul(transpose(conjg(proj(lo:hi,nao+1:2*nao))),qqc(1:nh,1:nh))+matmul(kk,c2(:,1:nh))
        do ip=1,3
          o1=(1+ip)*nao; o2=(4+ip)*nao
          faug(ip,j)=-2*real(sum(c1(:,1:nh)*transpose(proj(lo:hi,o1+1:o1+nao))) &
                            +sum(c2(:,1:nh)*transpose(proj(lo:hi,o2+1:o2+nao))),ofe_dp)
        enddo
      enddo
      deallocate(c1,c2,qqc,proj)
    endif
    deallocate(coef,z,x,grad,aa,kk,buf)
  end subroutine ofe_force_k

  ! ------------------------------------------------------------------ site operator in the AO basis
  subroutine ofe_dao(nao,nch,nshell,shell_l,shell_start,cloc,dch,mode,q,dq,use_x,clab,dao,ddao)
    ! D_ao = C_lab D C_lab^T with C_lab = Rblk(Q) C_loc.  mode: 0 global (Rblk = 1),
    ! 1 frozen (Q fixed, no derivative), 2 moving (derivatives w.r.t. anchors).
    ! ddao(:,:,k,p): derivative w.r.t. anchor p (1 z_from, 2 x_from, 3 origin) Cartesian k;
    ! use_x=.false. (constant helper x vector): no x_from derivative.
    integer, intent(in) :: nao, nch, nshell, shell_l(nshell), shell_start(nshell), mode
    real(ofe_dp), intent(in) :: cloc(nao,nch), q(3,3), dq(3,3,3,2)
    complex(ofe_dp), intent(in) :: dch(nch,nch)
    logical, intent(in) :: use_x
    real(ofe_dp), intent(out) :: clab(nao,nch)
    complex(ofe_dp), intent(out) :: dao(nao,nao), ddao(nao,nao,3,3)
    real(ofe_dp), allocatable :: rblk(:,:), drblk(:,:), dc(:,:)
    integer :: is, lo, nl, k, p
    allocate(rblk(nao,nao),drblk(nao,nao),dc(nao,nch))
    rblk=0
    if(mode==0) then
      do k=1,nao
        rblk(k,k)=1
      enddo
    else
      do is=1,nshell
        lo=shell_start(is); nl=ofe_nlm(shell_l(is))
        call ofe_rot_block(shell_l(is),q,rblk(lo:lo+nl-1,lo:lo+nl-1))
      enddo
    endif
    clab=matmul(rblk,cloc)
    dao=matmul(cmplx(clab,0._ofe_dp,ofe_dp),matmul(dch,cmplx(transpose(clab),0._ofe_dp,ofe_dp)))
    ddao=zero
    if(mode==2) then
      do p=1,2
        if(p==2 .and. .not.use_x) cycle
        do k=1,3
          drblk=0
          do is=1,nshell
            lo=shell_start(is); nl=ofe_nlm(shell_l(is))
            call ofe_drot_block(shell_l(is),q,dq(:,:,k,p),drblk(lo:lo+nl-1,lo:lo+nl-1))
          enddo
          dc=matmul(drblk,cloc)
          ddao(:,:,k,p)=matmul(cmplx(dc,0._ofe_dp,ofe_dp),matmul(dch,cmplx(transpose(clab),0._ofe_dp,ofe_dp))) &
                       +matmul(cmplx(clab,0._ofe_dp,ofe_dp),matmul(dch,cmplx(transpose(dc),0._ofe_dp,ofe_dp)))
        enddo
      enddo
      ddao(:,:,:,3)=-ddao(:,:,:,1)-ddao(:,:,:,2)
    endif
    deallocate(rblk,drblk,dc)
  end subroutine ofe_dao

  real(ofe_dp) function ofe_axial_err(nao,nch,nshell,shell_l,shell_start,cloc,dch) result(err)
    ! max |R D R^T - D| of the local AO operator D = C_loc D C_loc^T under rotations about local z
    ! (two angles).  A helper (atom-free) x axis is legal only if err ~ 0.
    integer, intent(in) :: nao, nch, nshell, shell_l(nshell), shell_start(nshell)
    real(ofe_dp), intent(in) :: cloc(nao,nch)
    complex(ofe_dp), intent(in) :: dch(nch,nch)
    real(ofe_dp) :: rblk(nao,nao), th(2)
    complex(ofe_dp) :: d(nao,nao), r(nao,nao)
    integer :: is, lo, nl, it
    th=[0.37_ofe_dp,1.13_ofe_dp]
    d=matmul(cmplx(cloc,0._ofe_dp,ofe_dp),matmul(dch,cmplx(transpose(cloc),0._ofe_dp,ofe_dp)))
    err=0
    do it=1,2
      rblk=0
      do is=1,nshell
        lo=shell_start(is); nl=ofe_nlm(shell_l(is))
        call ofe_rot_block(shell_l(is),ofe_rz(th(it)),rblk(lo:lo+nl-1,lo:lo+nl-1))
      enddo
      r=cmplx(rblk,0._ofe_dp,ofe_dp)
      err=max(err,maxval(abs(matmul(r,matmul(d,transpose(r)))-d)))
    enddo
  end function ofe_axial_err

  subroutine ofe_apply_stack(n,lda,m,ncol,w,dall,psi,hpsi,reduce)
    ! hpsi += W Dall W^H psi ; all sites stacked (ncol = sum of nch), one reduction.
    integer, intent(in) :: n, lda, m, ncol
    complex(ofe_dp), intent(in) :: w(lda,ncol), dall(ncol,ncol), psi(lda,m)
    complex(ofe_dp), intent(inout) :: hpsi(lda,m)
    procedure(ofe_reduce_iface) :: reduce
    complex(ofe_dp), allocatable :: cc(:,:), dc(:,:)
    allocate(cc(ncol,m),dc(ncol,m))
    call zgemm('C','N',ncol,m,n,one,w,lda,psi,lda,zero,cc,ncol)
    call reduce(cc)
    call zgemm('N','N',ncol,m,ncol,one,dall,ncol,cc,ncol,zero,dc,ncol)
    call zgemm('N','N',n,m,ncol,one,w,lda,dc,ncol,one,hpsi,lda)
    deallocate(cc,dc)
  end subroutine ofe_apply_stack
end module ofe_math
