;; -*- mode: lisp;-*-

(load "package://jsk_2014_picking_challenge/euslisp/utils.l")
(load "package://jsk_2014_picking_challenge/euslisp/robot-utils.l")

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_2014_picking_challenge")

;; -------------------
;; Get work order data
;; -------------------
(setq *work-order* (make-hash-table))
(defun work-order-cb (msg arm)
  (let (work-order bin object)
    (dolist (data (send msg :array))
      (setq bin (str-to-key (send data :bin)))
      (setq object (send data :object))
      (pushback (list bin object) work-order))
    (setf (gethash arm *work-order*) work-order)
    ))

(ros::subscribe "work_order/left_limb"
                jsk_2014_picking_challenge::WorkOrderArray
                #'(lambda (msg) (work-order-cb msg :larm)))
(ros::subscribe "work_order/right_limb"
                jsk_2014_picking_challenge::WorkOrderArray
                #'(lambda (msg) (work-order-cb msg :rarm)))
(defun get-work-order (arm)
  (ros::spin-once)
  (gethash arm *work-order*))

;; ---------------------
;; Get bin contents data
;; ---------------------
(setq *bin-contents* nil)
(defun bin-contents-cb (msg) (setq *bin-contents* (send msg :array)))

(ros::subscribe "/bin_contents" jsk_2014_picking_challenge::BinContentsArray
                #'bin-contents-cb)
(defun get-bin-contents (bin)
  (ros::spin-once)
  (dolist (bc *bin-contents*)
    (when (string= (key-to-str bin) (send bc :bin))
      (return-from get-bin-contents (send bc :objects)))))
