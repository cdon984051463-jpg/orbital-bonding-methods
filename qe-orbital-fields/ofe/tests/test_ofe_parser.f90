! Parser test (V1.2: namelist incl. print_hch, print_hch_k): runs the production parse_sites of orbital_field_engine.f90 (test copy with PRIVATE
! removed, built by run_cluster_tests.sh parser) on an ofe.in and dumps what it parsed; compared with the
! Python reader (tests/python/check_parser_dump.py).  No MPI, no QE state: parse_sites only uses errore on
! invalid input.
program test_ofe_parser
  use kinds, only: dp
  use orbital_field_engine
  implicit none
  character(len=linelen), allocatable :: lines(:)
  character(len=512) :: fname
  integer :: u, ios, nl, is, g, i, j, ip, sp
  namelist /ofe_control/ model, control, nsite, allow_relax, print_nao, dmax_eV, v_hard_eV, &
                         anchor_max_A, group_radius_A, fprint_min, history, print_hch, print_hch_k
  call get_command_argument(1,fname)
  allocate(lines(maxline)); lines=' '
  open(newunit=u,file=trim(fname),status='old',action='read')
  read(u,nml=ofe_control)
  nl=0
  do
    read(u,'(a)',iostat=ios) lines(nl+1)
    if(ios/=0) exit
    nl=nl+1
  enddo
  close(u)
  call parse_sites(nl,lines)
  write(*,'(a,i4,2l3,4es14.6)') 'CTL ',nsite,allow_relax,print_nao,dmax_eV,v_hard_eV,anchor_max_A,group_radius_A
  write(*,'(a,2l3)') 'CTL_HCH ',print_hch,print_hch_k
  do is=1,nsite
    associate(s=>site(is))
    write(*,'(a,i3,1x,a,1x,a,1x,a,1x,a,1x,a,4i5,2l3)') 'SITE ',is,trim(s%kind),trim(s%tag),trim(s%frame), &
      trim(s%center),trim(s%spin),s%natg,s%nao,s%nch,s%npar,s%cidentity,s%relax_ok
    do g=1,s%natg
      write(*,'(a,2i4,i6,3i4,1x,a,i3,4(1x,a))') 'ATOM ',is,g,s%atom(g),s%img(:,g),trim(s%elem(g)),s%nsh(g), &
        (trim(s%shlab(i,g)),i=1,s%nsh(g))
    enddo
    if(s%have_anchors) write(*,'(a,i4,3(i6,3i4),3(1x,a))') 'ANCH ',is,(s%anc_atom(i),s%anc_img(:,i),i=1,3), &
      (trim(s%anc_elem(i)),i=1,3)
    if(s%have_fref) write(*,'(a,i4,9es24.15)') 'FREF ',is,s%fref
    if(.not.s%cidentity) then
      do i=1,s%nao
        write(*,'(a,2i4,*(es24.15))') 'CLOC ',is,i,s%cloc(i,1:s%nch)
      enddo
    endif
    do sp=1,2
      if(.not.s%dread(sp)) cycle
      do i=1,s%nch
        write(*,'(a,3i4,*(es24.15))') 'DEV ',is,sp,i,(real(s%dch(i,j,sp),dp),aimag(s%dch(i,j,sp)),j=1,s%nch)
      enddo
    enddo
    do ip=1,s%npar
      write(*,'(a,2i4,1x,a,i3,es24.15,*(es24.15))') 'PAR ',is,ip,trim(s%pname(ip)),s%pspin(ip),s%pval(ip), &
        ((real(s%pb(i,j,ip),dp),aimag(s%pb(i,j,ip)),j=1,s%nch),i=1,s%nch)
    enddo
    end associate
  enddo
  write(*,'(a)') 'PARSE_OK'
end program test_ofe_parser
