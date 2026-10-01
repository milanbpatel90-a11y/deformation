"""Physical calibration and monotone region isolation for the real Gold basis.

All coordinates here are millimetres in the artist's native X/Y/Z frame.
The exporter remains responsible for metre conversion and rendering topology.
"""
import numpy as np


def smoothstep(value):
    t = np.clip(value, 0.0, 1.0)
    return t*t*(3.0-2.0*t)


def monotone_map(values, source, target, damping):
    """Interpolating warp with a strictly positive derivative and no overshoot.

    q(t)=(1-a)t+a(3t²-2t³), q'(t)=1-a+6at(1-t)>0 for a<1.
    Endpoints remain exact; outside the knots use positive linear continuation.
    """
    source, target = np.asarray(source), np.asarray(target)
    intervals = np.clip(np.searchsorted(source, values, side='right')-1, 0, len(source)-2)
    length = np.diff(source)[intervals]
    t = (values-source[intervals])/length
    a = np.asarray(damping)[intervals]
    q = (1-a)*t + a*smoothstep(t)
    # Continue with the endpoint derivative, rather than clamping coordinates.
    q = np.where(t < 0, (1-a)*t, np.where(t > 1, 1+(1-a)*(t-1), q))
    return target[intervals]+np.diff(target)[intervals]*q


class CalibratedRegions:
    names = ('frame_width', 'lens_width', 'lens_height', 'bridge_width')

    def __init__(self, rest, basis, slices, parameters, landmarks):
        self.rest, self.basis, self.slices = rest, basis, slices
        self.parameters = {p['name']: p for p in parameters}
        self.columns = [next(i for i,p in enumerate(parameters) if p['name']==name) for name in self.names]
        self.reference = self.measure(rest)
        # Differentiate the actual measured bounds, not nominal metadata values.
        step = 1e-4
        self.jacobian = np.column_stack([
            (self.measure(rest+basis[:,:,i]*step)-self.reference)/step for i in self.columns])
        if not np.isfinite(self.jacobian).all() or np.linalg.cond(self.jacobian) > 1e6:
            raise ValueError('Template dimension Jacobian is singular or ill-conditioned')
        self.calibration = np.linalg.inv(self.jacobian)
        frame, left, right = (rest[slices[n]] for n in ('Frame','LeftLens','RightLens'))
        self.x_source = np.array([frame[:,0].min(), right[:,0].min(), right[:,0].max(),
                                  left[:,0].min(), left[:,0].max(), frame[:,0].max()])
        self.z_source = np.array([left[:,2].min(),left[:,2].max()])
        self.temple_root = np.asarray(landmarks['LM_LeftTempleRoot'],dtype=float)
        self.temple_tip = np.asarray(landmarks['LM_LeftTempleTip'],dtype=float)
        self.temple_reference = float(np.linalg.norm(self.temple_tip-self.temple_root))
        # Protect the complete fixed hinge/front assembly. Inserts and arms
        # share the same longitudinal field, so the core cannot protrude.
        fixed = [rest[s] for name,s in slices.items() if not name.endswith(('Temple','FrameInsert'))]
        self.fixed_y = max(v[:,1].max() for v in fixed)
        # Use the full outer frame band for height falloff. Cutting this off
        # at the inner temple wall makes the field too curved for the artist's
        # long rim triangles and can fold their piecewise-linear approximation.
        self.height_falloff_x = max(abs(self.x_source[0]),abs(self.x_source[-1]))
        if np.any(np.diff(self.x_source)<=0) or self.height_falloff_x <= self.x_source[-2]:
            raise ValueError('Template optical regions are not ordered or isolated')

    def measure(self, vertices):
        frame,left,right = (vertices[self.slices[n]] for n in ('Frame','LeftLens','RightLens'))
        return np.array([np.ptp(frame[:,0]),np.ptp(left[:,0]),np.ptp(left[:,2]),
                         left[:,0].min()-right[:,0].max()])

    def configure(self, measurements):
        target = np.array([getattr(measurements,n) for n in self.names])
        self.coefficients = self.calibration @ (target-self.reference)
        # Apply calibrated NPZ vectors to the optical regions. These extrema
        # anchor the monotone field; the unsafe broad Frame falloffs are replaced.
        optical = {}
        for name in ('LeftLens','RightLens'):
            sl = self.slices[name]
            optical[name] = self.rest[sl]+self.basis[sl][:,:,self.columns] @ self.coefficients
        left,right = optical['LeftLens'],optical['RightLens']
        center = (self.x_source[0]+self.x_source[-1])/2
        optical_center = (self.x_source[2]+self.x_source[3])/2
        width = np.ptp(left[:,0])
        gap = left[:,0].min()-right[:,0].max()
        # Raw mirrored basis rounding must not translate the whole bridge when
        # lens width changes. Preserve its measured rest centre parametrically.
        self.x_target = np.array([center-target[0]/2,optical_center-gap/2-width,optical_center-gap/2,
                                   optical_center+gap/2,optical_center+gap/2+width,center+target[0]/2])
        if np.any(np.diff(self.x_target)<=0):
            raise ValueError('Calibrated optical regions overlap')
        self.z_target = np.array([left[:,2].min(),left[:,2].max()])
        self.z_scale = np.diff(self.z_target)[0]/np.diff(self.z_source)[0]
        slopes = np.diff(self.x_target)/np.diff(self.x_source)
        # Damping is LOCAL to each interval's strain. A global severity based
        # on temple length would subtly move the frame when only an arm changes.
        self.x_damping = 0.4*smoothstep(np.abs(slopes-1))
        self.x_damping[[1,3]] = 0  # Optical widths stay affine and exact.
        endpoints = self._xz(np.vstack([self.temple_root,self.temple_tip]))
        transverse = np.sum((endpoints[1,[0,2]]-endpoints[0,[0,2]])**2)
        remaining = measurements.temple_length**2-transverse
        if remaining <= 0:
            raise ValueError('Temple length cannot accommodate its transverse curve')
        self.target_tip_y = self.temple_root[1]+np.sqrt(remaining)
        y_scale = (self.target_tip_y-self.fixed_y)/(self.temple_tip[1]-self.fixed_y)
        if y_scale <= 0:
            raise ValueError('Temple tip would cross the fixed hinge assembly')
        self.y_damping = 0.4*float(smoothstep(abs(y_scale-1)))
        # The map is triangular: X=f(x), Z=g(x,z), Y=h(y). Positive
        # diagonal derivatives give a positive determinant everywhere.
        self.min_jacobian = float(np.min(slopes*(1-self.x_damping))*min(1,self.z_scale)*
                                  min(1,y_scale*(1-self.y_damping)))
        return self

    def _xz(self, points):
        out = np.array(points,dtype=float,copy=True)
        out[:,0] = monotone_map(points[:,0],self.x_source,self.x_target,self.x_damping)
        # Unit lens mask, zero in the bridge core and at the frame extremities;
        # smooth shared boundaries prevent cracks between frame and lenses.
        x = np.abs(points[:,0])
        inner = min(abs(self.x_source[2]),abs(self.x_source[3]))
        outer = max(abs(self.x_source[1]),abs(self.x_source[4]))
        weight = smoothstep((x-inner/2)/(inner/2))*(1-smoothstep((x-outer)/(self.height_falloff_x-outer)))
        desired_z = self.z_target[0]+(points[:,2]-self.z_source[0])*self.z_scale
        out[:,2] += weight*(desired_z-points[:,2])
        return out

    def warp(self, points):
        out = self._xz(points)
        moving = points[:,1]>self.fixed_y
        out[moving,1] = monotone_map(points[moving,1],
            [self.fixed_y,self.temple_tip[1]],[self.fixed_y,self.target_tip_y],[self.y_damping])
        return out
