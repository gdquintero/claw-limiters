! =====================================================
double precision function philim(a,b,meth,cfl)
! =====================================================
! Limitador phi(r), r = b/a, con el numero de Courant local cfl de la onda.
!   1-5      : limitadores clasicos de Clawpack (minmod, superbee, van Leer, MC, Beam-Warming)
!   21       : MINBEE (idea 1)       22 : hibrido original MINBEE-TOPUS (idea 2)
!   100+k    : hibrido H(delta,theta) = (1-w_delta) CFL-Superbee_theta + w_delta Arora-Roe_theta
!              con delta = hdelta(k), theta = htheta(k)  (common /limpar/)
!   200+k    : Arora-Roe con theta = htheta(k)
!   300+k    : CFL-Superbee con theta = htheta(k)
!   401      : beta-limitador de Kemm (ec. 43), beta = 2/3
    implicit double precision (a-h,o-z)
    common /limpar/ hdelta(20), htheta(20)
    double precision ubf, arf, csf, omf, minbeef, topusf

    r = b/a
    select case (meth)
    case (1) ! minmod
        philim = dmax1(0.d0, dmin1(1.d0, r))
    case (2) ! superbee
        philim = dmax1(0.d0, dmin1(1.d0, 2.d0*r), dmin1(2.d0, r))
    case (3) ! van Leer
        philim = (r + dabs(r)) / (1.d0 + dabs(r))
    case (4) ! monotonized centered
        c = (1.d0 + r)/2.d0
        philim = dmax1(0.d0, dmin1(c, 2.d0, 2.d0*r))
    case (5) ! Beam-Warming
        philim = r
    case (21)
        philim = minbeef(r)
    case (22)
        w = omf(r)
        philim = (1.d0 - w)*minbeef(r) + w*topusf(r)
    case (101:120)
        k = meth - 100
        w = omf(r)
        d2 = hdelta(k)**2
        w = d2*w / (1.d0 - w + d2*w)
        philim = (1.d0 - w)*csf(r, cfl, htheta(k)) + w*arf(r, cfl, htheta(k))
    case (201:220)
        philim = arf(r, cfl, htheta(meth - 200))
    case (301:320)
        philim = csf(r, cfl, htheta(meth - 300))
    case (401)
        xnu = dmin1(dmax1(cfl, 1.d-3), 1.d0 - 1.d-3)
        sl = (1.d0 + xnu)/3.d0
        bt = 2.d0/3.d0
        c = dmax1(1.d0 + (sl - bt/2.d0)*(r - 1.d0), 1.d0 + (sl + bt/2.d0)*(r - 1.d0))
        philim = dmax1(0.d0, dmin1(ubf(r, cfl, 1.d0), c))
    case default
        philim = 0.d0
    end select
    return
end function philim

double precision function ubf(r, cfl, th)
    implicit double precision (a-h,o-z)
    xnu = dmin1(dmax1(cfl, 1.d-3), 1.d0 - 1.d-3)
    ubf = dmax1(0.d0, dmin1(th*2.d0*r/xnu, th*2.d0/(1.d0 - xnu)))
end function ubf

double precision function arf(r, cfl, th)
    implicit double precision (a-h,o-z)
    double precision ubf
    xnu = dmin1(dmax1(cfl, 1.d-3), 1.d0 - 1.d-3)
    arf = dmax1(0.d0, dmin1(ubf(r, cfl, th), 1.d0 + (1.d0 + xnu)/3.d0*(r - 1.d0)))
end function arf

double precision function csf(r, cfl, th)
    implicit double precision (a-h,o-z)
    double precision ubf
    if (r > 0.d0) then
        csf = dmin1(ubf(r, cfl, th), dmax1(1.d0, r))
    else
        csf = 0.d0
    endif
end function csf

double precision function omf(r)
    implicit double precision (a-h,o-z)
    omf = (2.d0*dabs(r)/(r*r + 1.d0))**2
end function omf

double precision function minbeef(r)
    implicit double precision (a-h,o-z)
    minbeef = dmax1(0.d0, dmin1(1.5d0*r, 0.5d0*r + 0.5d0, 1.5d0))
end function minbeef

double precision function topusf(r)
    implicit double precision (a-h,o-z)
    if (r > 0.d0) then
        topusf = r*(6.d0*r + 2.d0)/(1.d0 + r)**3
    else
        topusf = 0.d0
    endif
end function topusf
