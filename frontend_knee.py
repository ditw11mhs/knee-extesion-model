import streamlit as st
import streamlit.components.v1 as components
from pandas import DataFrame as df
import plotly.express as px
import numpy as np
import os


class Main:
    def main(self):
        st.title("Knee Extension Simulation")
        with st.expander("Question"):
            self.question()
        with st.expander("Answer"):
            self.known()
            self.calculate_values()
            self.motion_equation()
        self.simulate()
    
    def simulate(self):
        camera_radius = st.sidebar.slider("Camera Radius", 0, 500, 190)
        camera_theta = st.sidebar.slider("Camera Theta", 0, 360, 90)
        camera_phi = st.sidebar.slider("Camera Phi", 0, 360, 0)
        H = 183
        l1 = 0.245 * H
        l2 = 0.246 * H
        l3 = 0.152 * H
        M = 93
        
        time = np.arange(0, 5, 0.001)
        f_shoulder = 0.25
        f_elbow = 0.25
        angle_x_shoulder = np.pi/2 * np.abs(np.sin(2 * np.pi * f_shoulder * time))
        angle_y_shoulder = angle_x_shoulder / 2
        angle_x_elbow = np.pi * np.abs(np.sin(2 * np.pi * f_elbow * time))
        angle_y_lower_arm = angle_x_elbow / 2

        plot_df = df(
            {
                "Time": time,
                "Shoulder X Angle": angle_x_shoulder,
                "Shoulder Y Angle": angle_y_shoulder,
                "Elbow X Angle": angle_x_elbow,
                "Lower Arm Y Angle": angle_y_lower_arm,
            }
        )
        plot_fig = px.line(
            plot_df,
            x="Time",
            y=[
                "Shoulder X Angle",
                "Shoulder Y Angle",
                "Elbow X Angle",
                "Lower Arm Y Angle",
            ],
        )
        
        components.html(
            f"""
            <canvas id="canvas" width = "640" height = "640"></canvas>
        
            
            <script src="https://twgljs.org/dist/3.x/twgl-full.min.js"></script>
            
            <script>
            const m4 = twgl.m4;
            const v3 = twgl.v3;
            const gl = document.querySelector("canvas").getContext("webgl");

            const vs = `
            attribute vec4 position;
            attribute vec3 normal;

            uniform mat4 u_projection;
            uniform mat4 u_view;
            uniform mat4 u_model;

            varying vec3 v_normal;

            void main() {{
            gl_Position = u_projection * u_view * u_model * position;
            v_normal = mat3(u_model) * normal; // better to use inverse-transpose-model
            }}
            `

            const fs = `
            precision mediump float;

            varying vec3 v_normal;

            uniform vec3 u_lightDir;
            uniform vec3 u_color;

            void main() {{
            float light = dot(normalize(v_normal), u_lightDir) * .5 + .5;
            gl_FragColor = vec4(u_color * light, 1);
            }}
            `;

            // compiles shaders, links program, looks up attributes
            const programInfo = twgl.createProgramInfo(gl, [vs, fs]);
            // calls gl.createBuffer, gl.bindBuffer, gl.bufferData
            
            sphereRad =3;
            
            const cubeBufferInfo = twgl.primitives.createCubeBufferInfo(gl, 1);
            const sphereBufferInfo = twgl.primitives.createSphereBufferInfo(gl,sphereRad,200,200);
            const cylinderBufferInfo = twgl.primitives.createCylinderBufferInfo(gl,sphereRad/2,{l1}+2*sphereRad,200,200);
            const cylinder2BufferInfo = twgl.primitives.createCylinderBufferInfo(gl,sphereRad/4,({l3}+2*sphereRad),200,200);
            const truncatCylinderBufferInfo = twgl.primitives.createTruncatedConeBufferInfo(gl,sphereRad/3,sphereRad/2,{l2}+2*sphereRad,200,200); 
            
            r = {camera_radius};
            theta_camera = {camera_theta}*Math.PI/180;
            phi = {camera_phi}*Math.PI/180;
            
            x = r*Math.sin(theta_camera)*Math.cos(phi);
            z = r*Math.sin(theta_camera)*Math.sin(phi);
            y = -r*Math.cos(theta_camera);
            
            
            const stack = [];

            const color = [1, 1,1];
            const lightDir = v3.normalize([x, y, z]);
            
            function render(time) {{
            time *= 0.001;
            f = 0.15;
            
            rotate_knee = -Math.PI/2*Math.abs(Math.sin(2*Math.PI*f*time));
            
            
            twgl.resizeCanvasToDisplaySize(gl.canvas);
            gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
            
            gl.enable(gl.DEPTH_TEST);
            gl.enable(gl.CULL_FACE);
            
            gl.useProgram(programInfo.program);
            
            const fov = Math.PI * .25;
            const aspect = gl.canvas.clientWidth / gl.canvas.clientHeight;
            const zNear = 0.01;
            const zFar = 1000;
            const projection = m4.perspective(fov, aspect, zNear, zFar);
            
            const cameraPosition = [x, y, z];
            const target = [0, 0, -({l1}/2+sphereRad+{l1/2}+sphereRad)];
            const up = [0, 1, 0];
            const camera = m4.lookAt(cameraPosition, target, up);
            
            const view = m4.inverse(camera);
            
            
            let m = m4.translation([0, 0, 0]);
            pushMatrix(m);
            {{
                m = m4.rotateX(m, Math.PI/2);
                drawSphere(projection, view, m);
                pushMatrix(m);
                {{
                    
                    m = m4.translate(m, [0, -({l1}/2+sphereRad), 0]);
                    drawCylinder(projection, view, m);
                    
                    pushMatrix(m);
                    {{
                    
                    m = m4.translate(m, [0, -({l1/2}+sphereRad), 0]);
                    m = m4.rotateX(m, rotate_knee);
                    
                    drawSphere(projection, view, m);
                    pushMatrix(m);
                    {{
                       
                        m = m4.translate(m, [0, -({l2/2}+sphereRad), 0]);
                        
                        drawCone(projection, view, m);
                        pushMatrix(m);
                        {{
                          
                            m = m4.translate(m, [0, -({l2/2}+sphereRad), 0]);
                            drawSphere(projection, view, m);
                            
                            m = m4.translate(m,[0,0,-({l3/2}+sphereRad)]);
                            m = m4.rotateX(m,Math.PI/2);
                            drawCylinder2(projection, view, m);
                            pushMatrix(m);
                        }}
                    }}
                    }}
                    
                }}
            
            }}
            
            m = popMatrix();

            requestAnimationFrame(render);
            }}
            requestAnimationFrame(render);

            function pushMatrix(m) {{
            stack.push(m);
            }}

            function popMatrix() {{
            return stack.pop();
            }}
            
            //function MotionEquation(thetaa,thetadota,phia,phidota){{
            //    thetadotdot=((Fmus+torque/180*pi)+m*sqr(length)/8*phidot*phidot*sin(theta)*cos(theta)-m*gravity*length/2*sin(theta))/(m*sqr(length)/4+inertia);
            //    phidotdot=(torque1-m*sqr(length)/4*phidot*thetadot*sin(theta)*cos(theta))/(m*sqr(length)/8);
            //    torque = -10*thetadot+6.1*exp(-5.9*(theta+10*pi/180))-10.5*exp(-21.8*(67*pi/180-theta));
            //    
            //}}
            
            //function rungekutta(thetab,thetadotb,phib,phidotb){{
            //    MotionEquation(thetab,thetadotb,phib,phidotb);
            //    k1=0.5*dt*thetadotdot;
            //    k11=0.5*dt*phidotdot;
            //  
            //    MotionEquation(thetab+0.5*dt*(thetadotb+0.5*k1),thetadotb+k1,phib+0.5*dt*(phidotb+0.5*k11),phidotb+k11);
            //    k2=0.5*dt*thetadotdot;
            //    k21=0.5*dt*phidotdot;
              
            //    MotionEquation(thetab+0.5*dt*(thetadotb+0.5*k1),thetadotb+k2,phib+0.5*dt*(phidotb+0.5*k11),phidotb+k21);
            //    k3=0.5*dt*thetadotdot;
            //    k31=0.5*dt*phidotdot;
              
             //   MotionEquation(thetab+dt*(thetadotb+k3),thetadotb+2*k3,phib+dt*(phidotb+k31),phidotb+2*k31);
             //   k4=0.5*dt*thetadotdot;
             //   k41=0.5*dt*phidotdot;
              
             //   theta=theta+dt*(thetadot+1/3*(k1+k2+k3));
             //   thetadot=thetadot+1/3*(k1+2*k2+2*k3+k4);
              
             //   phi=phi+dt*(phidot+1/3*(k11+k21+k31));
              //  phidot:=phidot+1/3*(k11+2*k21+2*k31+k41);
             //   }}
            
            function drawCube(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cubeBufferInfo);
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            
            twgl.drawBufferInfo(gl, cubeBufferInfo);
            }}
            
            function drawSphere(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, sphereBufferInfo);
        
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
        
            twgl.drawBufferInfo(gl, sphereBufferInfo);
            }}
            
            function drawCylinder(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cylinderBufferInfo);
        
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            
        
            twgl.drawBufferInfo(gl, cylinderBufferInfo);
            }}
            
            function drawCylinder2(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, cylinder2BufferInfo);
        
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
            
        
            twgl.drawBufferInfo(gl, cylinder2BufferInfo);
            }}
            
            function drawCone(projection, view, model) {{
            twgl.setBuffersAndAttributes(gl, programInfo, truncatCylinderBufferInfo);
        
            twgl.setUniforms(programInfo, {{
                u_color: color,
                u_lightDir: lightDir,
                u_projection: projection,
                u_view: view,
                u_model: model,
            }});
        
            twgl.drawBufferInfo(gl, truncatCylinderBufferInfo);
            }}
            
            </script>
            
            """,
            width=640,
            height=640,
        )

    def motion_equation(self):
        st.markdown("## Motion Equation")
        st.latex(
            r"""
             \begin{aligned}
             \bf{Position}                                                    \\
             x_1 & = a_1 sin\theta_1                                 \\
             y_1 & = -a_1 cos\theta_1                                \\\\
             x_2 & = l_1sin\theta_1+a_2 sin\theta_2                  \\
             y_2 & = -l_1cos\theta_1-a_2 cos\theta_2                 \\\\
             x_3 & = l_1sin\theta_1+l_2 sin\theta_2+a_3 sin\theta_3  \\
             y_3 & = -l_1cos\theta_1-l_2 cos\theta_2-a_3 cos\theta_3 \\\\
           \end{aligned}
           """
        )
        st.latex(
            r"""
             \begin{aligned}
             \bf{Velocity}\\
             \dot{x_1} & = a_1\dot{\theta_1}cos\theta_1                                                          \\
             \dot{y_1} & = a_1\dot{\theta_1}sin\theta_1                                                          \\\\
             \dot{x_2} & =l_1\dot{\theta_1}cos\theta_1+a_2\dot{\theta_2}cos\theta_2                              \\
             \dot{y_2} & =l_1\dot{\theta_1}sin\theta_1+a_2\dot{\theta_2}sin\theta_2                              \\\\
             \dot{x_3} & =l_1\dot{\theta_1}cos\theta_1+l_2\dot{\theta_2}cos\theta_2+a_3\dot{\theta_3}cos\theta_3 \\
             \dot{y_3} & =l_1\dot{\theta_1}sin\theta_1+l_2\dot{\theta_2}sin\theta_2+a_3\dot{\theta_3}sin\theta_3 \\\\
           \end{aligned}
           """
        )
        st.latex(
            r"""
             \begin{aligned}
             \bf{Velocity\;Squared}\\
             \dot{x_1}^2 & = a_1^{2}\dot{\theta_1}^{2}cos^{2}\theta_1                                                                                                                                                                \\
             \dot{y_1}^2 & = a_1^{2}\dot{\theta_1}^{2}sin^{2}\theta_1                                                                                                                                                                \\\\
             \dot{x_2}^2 & = l_1^{2}\dot{\theta_1}^{2}cos^{2}\theta_1+2a_2 l_1 \dot{\theta_1} \dot{\theta_2} cos \theta_1 cos \theta_2 + \dot{\theta_2}^{2}a_2^{2}cos^{2}\theta_2                                                    \\
             \dot{y_2}^2 & = l_1^{2}\dot{\theta_1}^{2}sin^{2}\theta_1+2a_2 l_1 \dot{\theta_1} \dot{\theta_2} sin \theta_1 sin \theta_2 + \dot{\theta_2}^{2}a_2^{2}sin^{2}\theta_2                                                    \\\\
             \dot{x_3}^2 & = l_1^{2}\dot{\theta_1}^{2}cos^{2}\theta_1  + \dot{\theta_2}^{2} l_2^{2}cos^{2}\theta_2+a_3^{2} \dot{\theta_3}^{2} cos^{2} \theta_3                                                                       \\
                         & + 2 \dot{\theta_2} l_2 a_3 \dot{\theta_3} cos \theta_2 cos \theta_3+ 2 l_2 l_1 \dot{\theta_1} \dot{\theta_2} cos \theta_1 cos \theta_2 +2 l_1 \dot{\theta_1} a_3 \dot{\theta_3} cos \theta_1 cos \theta_3 \\
             \dot{y_3}^2 & = l_1^{2}\dot{\theta_1}^{2}sin^{2}\theta_1  + \dot{\theta_2}^{2} l_2^{2}sin^{2}\theta_2+a_3^{2} \dot{\theta_3}^{2} sin^{2} \theta_3                                                                       \\
                         & + 2 \dot{\theta_2} l_2 a_3 \dot{\theta_3} sin \theta_2 sin \theta_3+ 2 l_2 l_1 \dot{\theta_1} \dot{\theta_2} sin \theta_1 sin \theta_2 +2 l_1 \dot{\theta_1} a_3 \dot{\theta_3} sin \theta_1 sin \theta_3 \\
           \end{aligned}
                 """
        )
        st.latex(
            r"""
                 \begin{aligned}
                 \bf{Resultant\;Velocity}\\
                 \dot{x_1}^2+ \dot{y_1}^2 & = a_1^{2}\dot{\theta_1}^{2}                                                                                                                                                                        \\
                 \dot{x_2}^2 +\dot{y_2}^2 & = l_1^2 \dot{\theta_1}^2 + 2 l_1 \dot{\theta_1} a_2 \dot{\theta_2} cos(\theta_1-\theta_2) +a_2^2 \dot{\theta_2}^2                                                                                  \\
                 \dot{x_3}^2 +\dot{y_3}^2 & = l_1^2 \dot{\theta_1}^2 +  l_2^2 \dot{\theta_2}^2 + a_3^2 \dot{\theta_3}^2                                                                                                                        \\
                                          & + 2 l_1 \dot{\theta_1} l_2 \dot{\theta_2} cos(\theta_1-\theta_2) + 2 l_1 \dot{\theta_1} a_3 \dot{\theta_3} cos(\theta_1-\theta_3) + 2 l_2 \dot{\theta_2} a_3 \dot{\theta_3} cos(\theta_2-\theta_3) \\
               \end{aligned}
                 """
        )
        st.latex(
            r"""
                 \begin{aligned}
                 \bf{Kinetic\;and\;Potential\;Energy}\\
                 EK_1  & =\frac{1}{2}m_1v_1^2+\frac{1}{2}I_1\dot{\theta_1}^{2}                                                      \\
                 EK_2  & =\frac{1}{2}m_2v_2^2+\frac{1}{2}I_2\dot{\theta_2}^{2}                                                      \\
                 EK_3  & =\frac{1}{2}m_3v_3^2+\frac{1}{2}I_3\dot{\theta_3}^{2}                                                      \\
                 EK    & =EK_1+EK_2+EK_3                                                                                            \\
                 EK    & =\frac{1}{2}(m_1v_1^2+m_2v_2^2+m_3v_3^2+I_1\dot{\theta_1}^{2}+I_2\dot{\theta_2}^{2}+I_3\dot{\theta_3}^{2}) \\
                 v_1^2 & =\dot{x_1}^2+\dot{y_1}^2                                                                                   \\
                 v_2^2 & =\dot{x_2}^2+\dot{y_2}^2                                                                                   \\
                 v_3^2 & =\dot{x_3}^2+\dot{y_3}^2                                                                                   \\\\
                 EP_1  & =m_1gh_1                                                                                                 \\
                 EP_2  & =m_2gh_2                                                                                                 \\
                 EP_3  & =m_3gh_3                                                                                                 \\
                 EP    & =EP_1+EP_2+EP_3                                                                                            \\
                 EP    & =g(m_1h_1+m_2h_2+m_3h_3)                                                                                   \\
                 h_1   & = l_1+l_2-a_1 cos\theta_1                                                                                  \\
                 h_2   & = l_2 -a_2 cos\theta_2                                                                                     \\
                 h_3   & = 0                                                                                                        \\
               \end{aligned}
                 """
        )
        st.latex(
            r"""
            \begin{aligned}
            \bf{Lagrange\;Function}\\
            L & = EK-EP                                                                                                                                                                                                                                                                \\
            L & =\frac{1}{2}(m_1v_1^2+m_2v_2^2+m_3v_3^2+I_1\dot{\theta_1}^{2}+I_2\dot{\theta_2}^{2}+I_3\dot{\theta_3}^{2})-g(m_1h_1+m_2h_2)                                                                                                                                            \\
            L & =\frac{1}{2}(m_1(a_1^{2}\dot{\theta_1}^{2})+m_2(l_1^2 \dot{\theta_1}^2 + 2 l_1 \dot{\theta_1} a_2 \dot{\theta_2} cos(\theta_1-\theta_2) +a_2^2 \dot{\theta_2}^2 )+m_3(l_1^2 \dot{\theta_1}^2 +  l_2^2 \dot{\theta_2}^2 + a_3^2 \dot{\theta_3}^2                        \\
              & + 2 l_1 \dot{\theta_1} l_2 \dot{\theta_2} cos(\theta_1-\theta_2) + 2 l_1 \dot{\theta_1} a_3 \dot{\theta_3} cos(\theta_1-\theta_3) + 2 l_2 \dot{\theta_2} a_3 \dot{\theta_3} cos(\theta_2-\theta_3))+I_1\dot{\theta_1}^{2}+I_2\dot{\theta_2}^{2}+I_3\dot{\theta_3}^{2}) \\
              & -g(m_1(l_1+l_2-a_1 cos\theta_1)+m_2(l_2 -a_2 cos\theta_2))                                                                                                                                                                                                                                                      \\
          \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \bf{Lagrange\;Equation}\\
            \frac{d}{dt} \frac{\partial L}{\partial \dot{\alpha}}-\frac{\partial L}{\partial \alpha}     & =\tau                                                                                                                                                                                                 \\
            \\
            \frac{\partial L}{\partial \theta_1}                                                         & =-m_2 l_1 \dot{\theta_1} a_2 \dot{\theta_2} sin(\theta_1-\theta_2)-m_3 (l_1 \dot{\theta_1} l_2 \dot{\theta_2} sin(\theta_1-\theta_2) + l_1 \dot{\theta_1} a_3 \dot{\theta_3} sin(\theta_1-\theta_3) ) \\
                                                                                                         & -g(m_1(a_1 sin\theta_1))                                                                                                                                                                              \\
            \frac{\partial L}{\partial \dot{\theta_1}}                                                   & =m_1a_1^{2}\dot{\theta_1}+m_2(l_1^2 \dot{\theta_1} +  l_1 a_2 \dot{\theta_2} cos(\theta_1-\theta_2) )+m_3(l_1^2 \dot{\theta_1}                                                                        \\
                                                                                                         & +  l_1 l_2 \dot{\theta_2} cos(\theta_1-\theta_2) +  l_1 a_3 \dot{\theta_3} cos(\theta_1-\theta_3) )+I_1\dot{\theta_1}                                                                                 \\
            \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta_1}}                                       & =m_1a_1^{2}\ddot{\theta_1}+m_2(l_1^2 \ddot{\theta_1} +  l_1 a_2 \ddot{\theta_2} cos(\theta_1-\theta_2) )+m_3(l_1^2 \ddot{\theta_1}                                                                    \\
                                                                                                         & +  l_1 l_2 \ddot{\theta_2} cos(\theta_1-\theta_2) +  l_1 a_3 \ddot{\theta_3} cos(\theta_1-\theta_3) )+I_1\ddot{\theta_1}                                                                              \\
            \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta_1}}-\frac{\partial L}{\partial \theta_1} & =\ddot{\theta_1}(m_1a_2^{2}+m_2l_1^{2}+m_3l_1^{2}+I_1)                                                                                                                                                \\
                                                                                                         & +\ddot{\theta_2}(m_2l_1a_2cos(\theta_1-\theta_2)+m_3l_1l_2cos(\theta_1-\theta_2))                                                                                                                     \\
                                                                                                         & +\ddot{\theta_3}(m_3l_1a_3cos(\theta_1-\theta_3))                                                                                                                                                     \\
                                                                                                         & +\dot{\theta_1}(m_2l_1a_2\dot{\theta_2}sin(\theta_1-\theta_2))                                                                                                                                        \\
                                                                                                         & +\dot{\theta_2}(m_3l_1l_2\dot{\theta_1}sin(\theta_1-\theta_2))                                                                                                                                        \\
                                                                                                         & +\dot{\theta_3}(m_3l_1a_3\dot{\theta_1}sin(\theta_1-\theta_3))                                                                                                                                        \\
                                                                                                         & +gm_1a_2sin\theta_1                                                                                                                                                                                   \\
          \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \frac{\partial L}{\partial \theta_2}                                                         & =m_2 a_2 \dot{\theta_1} l_1 \dot{\theta_2} sin(\theta_1-\theta_2)+ m_3 \dot{\theta_1} l_1 \dot{\theta_2} l_2 sin(\theta_1-\theta_2) - m_3 a_3 \dot{\theta_3} l_2 \dot{\theta_2} sin(\theta_2-\theta_3))-a_2 g m_2 sin(\theta_2) \\
            \frac{\partial L}{\partial \dot{\theta_2}}                                                   & = \dot{\theta_2}(m_2 a_2^{2}+ m_3 l_2^{2})+m_2 \dot{\theta_1} l_1 a_2 cos(\theta_1-\theta_2)+ m_3 \dot{\theta_1} l_1 l_2 cos(\theta_1-\theta_2) + m_3 a_3 \dot{\theta_3} l_2 cos(\theta_2-\theta_3) + I_2 \dot{\theta_2}        \\
            \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta_2}}                                       & = \ddot{\theta_2}(m_2 a_2^{2}+ m_3 l_2^{2})+m_2 \ddot{\theta_1} l_1 a_2 cos(\theta_1-\theta_2)+ m_3 \ddot{\theta_1} l_1 l_2 cos(\theta_1-\theta_2) + m_3 a_3 \ddot{\theta_3} l_2 cos(\theta_2-\theta_3) + I_2 \ddot{\theta_2}   \\
            \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta_2}}-\frac{\partial L}{\partial \theta_2} & =\ddot{\theta_1}(m_2l_1a_2cos(\theta_1-\theta_2)+m_3l_1l_2cos(\theta_1-\theta_2))                                                                                                                                               \\
                                                                                                         & +\ddot{\theta_2}(m_2a_2^{2} + m_3l_2^{2} +I_2)                                                                                                                                                                                  \\
                                                                                                         & +\ddot{\theta_3}(m_3a_3l_2cos(\theta_2-\theta_3))                                                                                                                                                                               \\
                                                                                                         & +\dot{\theta_1}(-m_2l_1a_2\dot{\theta_2}sin(\theta_1-\theta_2))                                                                                                                                                                 \\
                                                                                                         & +\dot{\theta_2}(-m_3\dot{\theta_1}l_1l_2sin(\theta_1-\theta_2))                                                                                                                                                                 \\
                                                                                                         & +\dot{\theta_3}(m_3a_3l_2\dot{\theta_2}sin(\theta_2-\theta_3))                                                                                                                                                                  \\
                                                                                                         & +a_2gm_2sin\theta_2
          \end{aligned}
            """
        )
        st.latex(
            r"""
            \begin{aligned}
            \frac{\partial L}{\partial \theta_3}                                                         & = m_3 a_3 \dot{\theta_3} ( l_1 \dot{\theta_1} sin(\theta_1-\theta_3) + l_2 \dot{\theta_2} sin(\theta_2-\theta_3))                                     \\
            \frac{\partial L}{\partial \dot{\theta_3}}                                                   & = m_3l_3^{2}\dot{\theta_3} +  m_3 a_3 ( l_1 \dot{\theta_1} cos(\theta_1-\theta_3)+ l_2 \dot{\theta_2} cos(\theta_2-\theta_3)) +I_3 \dot{\theta_3}     \\
            \frac{d}{dt}\frac{\partial L}{\partial \dot{\theta_3}}                                       & = m_3l_3^{2}\ddot{\theta_3} +  m_3 a_3 ( l_1 \ddot{\theta_1} cos(\theta_1-\theta_3)+ l_2 \ddot{\theta_2} cos(\theta_2-\theta_3)) +I_3 \ddot{\theta_3} \\
            \frac{d}{dt} \frac{\partial L}{\partial \dot{\theta_3}}-\frac{\partial L}{\partial \theta_3} & =\ddot{\theta_1}(m_3a_3l_1cos(\theta_1-\theta_3))                                                                                                     \\
                                                                                                         & +\ddot{\theta_2}(m_3a_3l_2cos(\theta_2-\theta_3))                                                                                                     \\
                                                                                                         & +\ddot{\theta_3}(m_3l_3^2 + I_3)                                                                                                                      \\
                                                                                                         & +\dot{\theta_1}(m_3a_3\dot{\theta_3}l_1sin(\theta_1-\theta_3))                                                                                        \\
                                                                                                         & +\dot{\theta_2}(m_3a_3\dot{\theta_3}l_2sin(\theta_2-\theta_3))                                                                                        \\
                                                                                                         & +\dot{\theta_3}(0)                                                                                                                                    \\
                                                                                                         & +0
          \end{aligned}
            """
        )
        st.image(os.path.join("image","tau_table.png"))
        st.image(os.path.join("image","tau_eq.png"))
        st.latex(
            r"""
            \begin{aligned}
            \bf{Matrix\;Motion\;Equations}\\
            \bf{M}\ddot{\theta} + \bf{C}\dot{\theta} + \bf{G} & = \bf{\tau}                     \\
            \ddot{\theta}                                     & = \begin{bmatrix}
              \ddot{\theta_1} \\
              \ddot{\theta_2} \\
              \ddot{\theta_3}
            \end{bmatrix}\;\;
            \dot{\theta} = \begin{bmatrix}
              \dot{\theta_1} \\
              \dot{\theta_2} \\
              \dot{\theta_3}
            \end{bmatrix}\\
            M                                                 & =\begin{bmatrix}
              m_1a_2^{2}+m_2l_1^{2}+m_3l_1^{2}+I_1                            & m_2l_1a_2cos(\theta_1-\theta_2)+m_3l_1l_2cos(\theta_1-\theta_2) & m_3l_1a_3cos(\theta_1-\theta_3) \\
              m_2l_1a_2cos(\theta_1-\theta_2)+m_3l_1l_2cos(\theta_1-\theta_2) & m_2a_2^{2} + m_3l_2^{2} +I_2                                    & m_3a_3l_2cos(\theta_2-\theta_3) \\
              m_3a_3l_1cos(\theta_1-\theta_3)                                 & m_3a_3l_2cos(\theta_2-\theta_3)                                 & m_3l_3^2 + I_3
            \end{bmatrix}      \\
            C                                                 & = \begin{bmatrix}
              m_2l_1a_2\dot{\theta_2}sin(\theta_1-\theta_2)  & m_3l_1l_2\dot{\theta_1}sin(\theta_1-\theta_2)  & m_3l_1a_3\dot{\theta_1}sin(\theta_1-\theta_3) \\
              -m_2l_1a_2\dot{\theta_2}sin(\theta_1-\theta_2) & -m_3\dot{\theta_1}l_1l_2sin(\theta_1-\theta_2) & m_3a_3l_2\dot{\theta_2}sin(\theta_2-\theta_3) \\
              m_3a_3\dot{\theta_3}l_1sin(\theta_1-\theta_3)  & m_3a_3\dot{\theta_3}l_2sin(\theta_2-\theta_3)  & 0
            \end{bmatrix}     \\
            G                                                 & = \begin{bmatrix}
              gm_1a_2sin\theta_1 \\
              gm_2a_2sin\theta_2 \\
              0
            \end{bmatrix}     \\
            \tau                                              & = \begin{bmatrix}
              \tau_p1 \\
              \tau_p2 \\
              \tau_p3
            \end{bmatrix}     \\
                                                              & =\begin{bmatrix}
              -3.09\dot{\theta_1}+2.6e^{-5.8(\theta_1+10^{o})}+8.7e^{1.3(\theta_1-10^{o})} \\
              -10\dot{\theta_2}+6.1e^{-5.9(\theta_2-10^{o})}+10.5e^{21.8(\theta_1-67^{o})} \\
              -0.943\dot{\theta_3}+2e^{-5(\theta_3+15^{o})}+2e^{5(\theta_1-25^{o})}
            \end{bmatrix}      \\
          \end{aligned}
            """
        )

    def calculate_values(self):
        st.markdown(
            """
                        Using the above figure, we can calculate the following values:     
            """
        )
        st.latex(
            r"""
                \begin{aligned}
                l_1 & = Thigh\;Length                                           \\
                l_2 & = Shank\;Length                                           \\
                l_3 & = Foot\;Length                                            \\
                a_1 & = Thigh\;Proximal\;Length                                 \\
                a_2 & = Shank\;Proximal\;Length                                 \\
                a_3 & = Foot\;Proximal\;Length                                  \\
                m_1 & = Thigh\;Mass                                             \\
                m_2 & = Shank\;Mass                                             \\
                m_3 & = Foot\;Mass                                              \\
                I_1 & = Thigh\;Inertia                                          \\
                I_2 & = Shank\;Inertia                                          \\
                I_3 & = Foot\;Inertia                                           \\\\
              
                l_1 & = 0.245H                                                  \\
                l_2 & = 0.246H                                                  \\
                l_3 & = 0.152H                                                  \\
                a_1 & =0.433l_1                                                 \\
                a_2 & =0.433l_2                                                 \\
                a_3 & =0.5l_3                                                   \\
                m_1 & = 0.1M                                                    \\
                m_2 & =0.0465M                                                  \\
                m_3 & =0.0145M                                                  \\
                I_1 & =m_1(l_1\times 0.323)^2                                   \\
                I_2 & =m_2(l_2\times 0.302)^2                                   \\
                I_3 & =m_3(l_3\times 0.475)^2                                   \\\\
              
                H   & = 180\;cm=1.8m                                            \\
                M   & = 93\;kg                                                  \\\\
              
                l_1 & = 0.245\times 1.8 =   0.441\;m                            \\
                l_2 & = 0.246\times 1.8 =     0.4428\;m                         \\
                l_3 & = 0.152\times 1.8 =    0.2736    \;m                      \\
                a_1 & =0.433\times 0.441 = 0.190953\;m                          \\
                a_2 & =0.433\times 0.4428 =      0.1917324   \;m                \\
                a_3 & =0.5\times 0.2736 =    0.1368\;m                          \\
                m_1 & = 0.1\times  93 = 9.3\;kg                                 \\
                m_2 & =0.0465\times 93 =  4.3245\;kg                            \\
                m_3 & =0.0145\times 93 = 1.3485\;kg                             \\
                I_1 & =9.3(0.441\times 0.323)^2 =0.18869707671570005\;kgm^2     \\
                I_2 & =4.3245(0.4428\times 0.302)^2=0.07733302734438431 \;kgm^2 \\
                I_3 & =1.3485(0.2736\times 0.475)^2=0.0227756277576\;kgm^2      \\\\
              \end{aligned}
                 """
        )

    def known(self):
        st.markdown("## Anthropometric Calculation")
        st.image(
            os.path.join("image", "body_segment.png"),
            caption="Body segment lengths expressed as a fraction of body height H . Source: David A. Winter - Biomechanics and Motor Control of Human Movement",
            width=450,
        )
        st.image(
            os.path.join("image", "anthropometric_data.png"),
            caption="""Source: M, Dempster via Miller and Nelson; Biomechanics of Sport, Lea and Febiger, Philadelphia, 1973. P, Dempster via Plagenhoef; Patterns of
            Human Motion, Prentice-Hall, Inc. Englewood Cliffs, NJ, 1971. L, Dempster via Plagenhoef from living subjects; Patterns of Human Motion, Prentice-Hall,
            Inc., Englewood Cliffs, NJ, 1971. C, Calculated.""",
        )

    def question(self):
        st.image(os.path.join("image", "three_limb_model.png"))
        st.markdown(
            """
                        Use a geometric description of three joint of limb limb model shown in above figure to derive motion 
                        equations based on the Lagrangian Method. Define all active and passive torque of each joint, all 
                        skeletal model parameter values from computational biomechanics data. Use the BW and BH of each 
                        student as data model, and define segmental data based on the regression of human anthropometric 
                        data. Arrange the complete model in matrix form.
                        From the model you derived, with focusing on knee joint movement, realize knee extension computer 
                        model movement simulation. Use this equation for recruitment curve l_1:= 0.5xtanh(15.0x(s1-
                        0.5))+0.5, tr=100 ms, tf=150 ms for activation dynamics.
                        """
        )
        st.image(os.path.join("image", "block_diagram.png"))


if __name__ == "__main__":
    main = Main()
    main.main()
