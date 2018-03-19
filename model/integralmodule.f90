!Fortran module containing subroutines to perform numerical quadrature on 
!the 2-sphere using the quadrature rules given in quadrature_rules.py 


module integralRoutines 

use omp_lib
implicit none

!Some parameters
integer, parameter :: dp = selected_real_kind(15)
real(dp), parameter :: pi = 4.D0*DATAN(1.D0)

contains


function dist(phi1, theta1, phi2, theta2, radius)
    !Function to compute great circle distance

    real(kind=8), intent(in) :: phi1, theta1, phi2, theta2, radius
    real(kind=8) :: dist, a

    a = cos(phi1)*cos(phi2) + sin(phi1)*sin(phi2)*cos(theta1 - theta2)
    dist = radius*acos(a)

end function dist


subroutine make_Kernel(n, p, theta, phi, kern)
    !Function to compute the connectivity matrix 

    integer, intent(in) :: n
    real(kind=8), dimension(10), intent(in) :: p
    real(kind=8), dimension(n), intent(in) :: theta, phi
    real(kind=8), dimension(n,n), intent(out) :: kern
    real(kind=8) :: d, connect
    real(kind=8) :: h, mu, gain, a1, b1, a2, b2, radius, amp
    integer :: i, j
    
    !Unpack parameters
    h = p(2)
    mu = p(3)
    gain = p(4)
    a1 = p(5)
    b1 = p(6)
    a2 = p(7)
    b2 = p(8)
    radius = p(9)
    amp = p(10)
    
    !Compute kernel matrix
    kern = 0.0_dp
    !$OMP PARALLEL DO num_threads(2)
    do i=1,n
        do j=1,n
        
            !Compute great circle disstance
            d = dist(phi(i), theta(i), phi(j), theta(j), radius)
            if (i == j) then
                d = 0.0_dp
            end if
            
            !Compute connectivity
            kern(i,j) = a1*exp(-d*d/b1) - a2*exp(-d*d/b2)      
        end do
    end do
    !$OMP END PARALLEL DO
        
end subroutine make_Kernel


subroutine make_F(n, p, theta, phi, kern, w, u, F)
    !Function to compute the integral operator for the system
    !using a quadrature rule  

    integer, intent(in) :: n
    real(kind=8), dimension(10), intent(in) :: p
    real(kind=8), dimension(n,n), intent(in) :: kern
    real(kind=8), dimension(n), intent(in) :: theta, phi, w, u
    real(kind=8), dimension(n), intent(out) :: F
    real(kind=8), dimension(n) :: s
    real(kind=8) :: h, mu, gain 
    integer :: i, j

    !Unpack parameters
    h = p(2)
    mu = p(3)
    gain = p(4)

    !Initialize things
    F = 0.0_dp
    s = 1.0_dp/(1.0_dp + exp(-mu*(u - h)))

    !Compute the integral by quadrature
    !$OMP PARALLEL DO num_threads(2)
    do i=1,n
        do j=1,n            
            F(i) = F(i)  +  w(j)*kern(i,j)*s(j)
        end do
    end do
    !$OMP END PARALLEL DO

    !Construct F
    F = -u + gain*F

end subroutine make_F


subroutine make_Jv(n, p, theta, phi, kern, w, u, v, Jv)

    integer, intent(in) :: n
    real(kind=8), dimension(10), intent(in) :: p
    real(kind=8), dimension(n,n), intent(in) :: kern
    real(kind=8), dimension(n), intent(in) :: theta, phi, w, u, v
    real(kind=8), dimension(n), intent(out) :: Jv
    real(kind=8), dimension(n) :: ds
    real(kind=8) :: h, mu, gain 
    integer :: i, j

    !Unpack parameters
    h = p(2)
    mu = p(3)
    gain = p(4)

    !Initialize things
    Jv = 0.0_dp
    
    !Compute derivative of firing rate
    ds = 1.0_dp/(1.0_dp + exp(-mu*(u-h)))
    ds = mu*ds*(1.0_dp - ds)
    ds = v*ds 

    !Compute the integral by quadrature
    !$OMP PARALLEL DO num_threads(2)
    do i=1,n
        do j=1,n             
            Jv(i) = Jv(i)  +  w(j)*kern(i,j)*ds(j)
        end do
    end do
    !$OMP END PARALLEL DO

    !Construct F
    Jv = -v + gain*Jv

end subroutine make_Jv

end module
