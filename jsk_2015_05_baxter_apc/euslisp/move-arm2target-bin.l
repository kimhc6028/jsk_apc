#!/usr/bin/env roseus

(load "package://baxtereus/baxter-interface.l")
(load "package://jsk_2014_picking_challenge/euslisp/target-pick-region.l")

(ros::roseus "move2target_listener")
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "std_msgs")

#|
(defclass target-pick-region
  :super propertied-object
  :slots (a b c d e f g h i j k l))
(defmethod target-pick-region
  (:init ()
    """Local coordinates of each region
    in the podlowres objects"""
    (setq a #f(-270 1660 500))
    (setq b #f(   0 1660 500))
    (setq c #f( 270 1660 500))
    (setq d #f(-270 1430 500))
    (setq e #f(   0 1430 500))
    (setq f #f( 270 1430 500))
    (setq g #f(-270 1200 500))
    (setq h #f(   0 1200 500))
    (setq i #f( 270 1200 500))
    (setq j #f(-270  950 500))
    (setq k #f(   0  950 500))
    (setq l #f( 270  950 500)))
  (:target-list ()
    (list a b c d e f g h i j k l))
  (:target-name-list ()
    '(a b c d e f g h i j k l)))
|#

(defun init ()
  (baxter-init)
  (send *baxter* :locate #f(0 0 950) :world)
  (when (not (boundp '*pod*))
    (load "package://jsk_2014_picking_challenge/euslisp/pod-lowres.l")
    (send *pod* :locate #f(1300 0 0) :world)
    (send *pod* :rotate pi/2 :x :world)
    (send *pod* :rotate (- pi/2) :z :world)
    )
  (objects (list *baxter* *pod*)))

(defun subscribe-init ()
  (ros::subscribe "/semi/move_larm2target" std_msgs::String #'cb-larm)
  (ros::subscribe "/semi/move_rarm2target" std_msgs::String #'cb-rarm)
  (ros::ros-info "subscriber initialized"))

(defun cb-larm (msg)
  (callback msg ':larm))

(defun cb-rarm (msg)
  (callback msg ':rarm))

(defun callback (msg arm)
  """msg data should be string and included in (a, b, c, ..., l)"""
  (let ((region (instance target-pick-region :init))
        nm c cw target)
    ; (setq region (instance target-pick-region :init))
    (setq nm (intern (string-upcase (send msg :data))))
    (ros::ros-info (format nil "moving to ~A." nm))
    ;; target data validation
    (unless (find nm (send region :target-name-list))
      (ros::ros-info (format nil "~A not found." nm))
      (return-from callback nil))
    ;; get target coordinates from target name
    (setq c (send region :get-val nm))
    ;; transform target coordinates local->world
    (setq cw (send *pod* :transform-vector c))
    ;; ik with z axis constraction
    (ros::ros-info (format nil "computing IK for ~A." nm))
    (setq target (instance cascaded-coords :init :pos cw))
    (send *baxter* arm :inverse-kinematics target
          :rotate-axis :z)
    ;; reflect change to ri
    (send *ri* :angle-vector (send *baxter* :angle-vector))
    (send *irtviewer* :draw-objects)
    (x::window-main-one)
    ))

(init)
(subscribe-init)
(ros::spin)
